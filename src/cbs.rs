use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use aligned_vec::ABox;
use refined_tfhe_lhe::int_lhe_params::IntLheParam;
use refined_tfhe_lhe::{
    circuit_bootstrap_lwe_ciphertext_by_trace_with_preprocessing, convert_lwe_to_glwe_const,
    gen_all_auto_keys, gen_blind_rotate_local_assign, generate_scheme_switching_key,
    glwe_ciphertext_clone_from, glwe_ciphertext_monic_monomial_div_assign, keygen_pbs,
    lwe_preprocessing_assign, polynomial_mul_by_fft, switch_scheme, trace_assign, AutomorphKey,
};
use tfhe::core_crypto::{
    algorithms::polynomial_algorithms::polynomial_wrapping_monic_monomial_mul,
    commons::dispersion::DispersionParameter,
    commons::{generators::DeterministicSeeder, math::random::Seed},
    fft_impl::fft64::{
        c64,
        crypto::{
            bootstrap::FourierLweBootstrapKeyOwned,
            ggsw::{FourierGgswCiphertext, FourierGgswCiphertextList},
        },
        math::fft::Fft,
    },
    prelude::*,
};

use crate::ms_noise_reduction::{
    improve_lwe_ciphertext_modulus_switch_noise_for_binary_key, NoiseEstimationMeasureBound,
    RSigmaFactor,
};

pub type SelectorCiphertext = FourierGgswCiphertext<ABox<[c64]>>;
pub type SchemeSwitchKey = FourierGgswCiphertextList<Vec<c64>>;
pub const FUSED_SELECTOR_GROUP_BITS: usize = 3;
const FUSED_SELECTOR_LUT_COUNT_LOG: usize = 4;
const FUSED_SELECTOR_LUT_COUNT: usize = 1 << FUSED_SELECTOR_LUT_COUNT_LOG;
const DENSE_STANDARD_BR_SLOT_START: usize =
    FUSED_SELECTOR_LUT_COUNT - (1 << FUSED_SELECTOR_GROUP_BITS);
const INTER_GROUP_STANDARD_BR_SLOT_START: usize = 1;
const INTER_GROUP_STANDARD_BR_SLOT_STRIDE: usize = 2;
const STANDARD_BR_INPUT_BIAS: i128 = 0;
const STANDARD_BR_LINEAR_BOOLEAN_MU: u64 = 1u64 << 58;
const STANDARD_BR_LINEAR_GROUP_SCALE: u64 = 2;
const STANDARD_BR_LINEAR_SLOT_COUNT: usize = 8;
const STANDARD_BR_LINEAR_SLOT_OFFSET_UNITS: u64 = 8;
const MS_NOISE_REDUCTION_ZERO_ENCRYPTION_COUNT: usize = 128;
const MS_NOISE_REDUCTION_R_SIGMA: f64 = 13.0;
const MS_NOISE_REDUCTION_MEASURE_BOUND: f64 = 0.0;

pub struct CircuitBootstrapKeys {
    pub params: IntLheParam<u64>,
    pub large_lwe_secret_key: LweSecretKey<Vec<u64>>,
    pub glwe_secret_key: GlweSecretKey<Vec<u64>>,
    pub small_lwe_secret_key: LweSecretKey<Vec<u64>>,
    small_lwe_zero_encryptions: Vec<LweCiphertext<Vec<u64>>>,
    pub fourier_bsk: FourierLweBootstrapKeyOwned,
    pub ksk_large_to_small: LweKeyswitchKey<Vec<u64>>,
    pub auto_keys: HashMap<usize, AutomorphKey<ABox<[c64]>>>,
    pub scheme_switch_key: SchemeSwitchKey,
    deterministic_seed: Option<u128>,
    next_seed: AtomicU64,
}

fn generate_small_lwe_zero_encryptions(
    params: &IntLheParam<u64>,
    small_lwe_secret_key: &LweSecretKey<Vec<u64>>,
    encryption_generator: &mut EncryptionRandomGenerator<ActivatedRandomGenerator>,
) -> Vec<LweCiphertext<Vec<u64>>> {
    (0..MS_NOISE_REDUCTION_ZERO_ENCRYPTION_COUNT)
        .map(|_| {
            allocate_and_encrypt_new_lwe_ciphertext(
                small_lwe_secret_key,
                Plaintext(0u64),
                params.lwe_modular_std_dev(),
                params.ciphertext_modulus(),
                encryption_generator,
            )
        })
        .collect()
}

impl CircuitBootstrapKeys {
    pub fn deterministic_seed(&self) -> Option<u128> {
        self.deterministic_seed
    }

    pub fn new(params: IntLheParam<u64>) -> Self {
        Self::new_internal(params, None)
    }

    pub fn new_with_seed(params: IntLheParam<u64>, seed: u128) -> Self {
        Self::new_internal(params, Some(seed))
    }

    pub fn new_with_shared_glwe_secret(
        params: IntLheParam<u64>,
        glwe_secret_key: &GlweSecretKey<Vec<u64>>,
    ) -> Self {
        Self::new_with_shared_glwe_secret_internal(params, glwe_secret_key, None)
    }

    pub fn new_with_shared_glwe_secret_and_seed(
        params: IntLheParam<u64>,
        glwe_secret_key: &GlweSecretKey<Vec<u64>>,
        seed: u128,
    ) -> Self {
        Self::new_with_shared_glwe_secret_internal(params, glwe_secret_key, Some(seed))
    }

    fn new_internal(params: IntLheParam<u64>, deterministic_seed: Option<u128>) -> Self {
        let (
            large_lwe_secret_key,
            glwe_secret_key,
            small_lwe_secret_key,
            small_lwe_zero_encryptions,
            fourier_bsk,
            ksk,
            auto_keys,
            scheme_switch_key,
        ) = if let Some(seed) = deterministic_seed {
            let mut seeder = DeterministicSeeder::<ActivatedRandomGenerator>::new(Seed(seed));
            let mut secret_generator =
                SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());
            let mut encryption_generator =
                EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(
                    seeder.seed(),
                    &mut seeder,
                );

            let (large_lwe_secret_key, glwe_secret_key, small_lwe_secret_key, fourier_bsk, ksk) =
                keygen_pbs(
                    params.lwe_dimension(),
                    params.glwe_dimension(),
                    params.polynomial_size(),
                    params.lwe_modular_std_dev(),
                    params.glwe_modular_std_dev(),
                    params.pbs_base_log(),
                    params.pbs_level(),
                    params.ks_base_log(),
                    params.ks_level(),
                    &mut secret_generator,
                    &mut encryption_generator,
                );

            let auto_keys = gen_all_auto_keys(
                params.auto_base_log(),
                params.auto_level(),
                params.fft_type_auto(),
                &glwe_secret_key,
                params.glwe_modular_std_dev(),
                &mut encryption_generator,
            );

            let scheme_switch_key = generate_scheme_switching_key(
                &glwe_secret_key,
                params.ss_base_log(),
                params.ss_level(),
                params.glwe_modular_std_dev(),
                params.ciphertext_modulus(),
                &mut encryption_generator,
            );

            let small_lwe_zero_encryptions = generate_small_lwe_zero_encryptions(
                &params,
                &small_lwe_secret_key,
                &mut encryption_generator,
            );

            (
                large_lwe_secret_key,
                glwe_secret_key,
                small_lwe_secret_key,
                small_lwe_zero_encryptions,
                fourier_bsk,
                ksk,
                auto_keys,
                scheme_switch_key,
            )
        } else {
            let mut boxed_seeder = new_seeder();
            let seeder = boxed_seeder.as_mut();
            let mut secret_generator =
                SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());
            let mut encryption_generator =
                EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed(), seeder);

            let (large_lwe_secret_key, glwe_secret_key, small_lwe_secret_key, fourier_bsk, ksk) =
                keygen_pbs(
                    params.lwe_dimension(),
                    params.glwe_dimension(),
                    params.polynomial_size(),
                    params.lwe_modular_std_dev(),
                    params.glwe_modular_std_dev(),
                    params.pbs_base_log(),
                    params.pbs_level(),
                    params.ks_base_log(),
                    params.ks_level(),
                    &mut secret_generator,
                    &mut encryption_generator,
                );

            let auto_keys = gen_all_auto_keys(
                params.auto_base_log(),
                params.auto_level(),
                params.fft_type_auto(),
                &glwe_secret_key,
                params.glwe_modular_std_dev(),
                &mut encryption_generator,
            );

            let scheme_switch_key = generate_scheme_switching_key(
                &glwe_secret_key,
                params.ss_base_log(),
                params.ss_level(),
                params.glwe_modular_std_dev(),
                params.ciphertext_modulus(),
                &mut encryption_generator,
            );

            let small_lwe_zero_encryptions = generate_small_lwe_zero_encryptions(
                &params,
                &small_lwe_secret_key,
                &mut encryption_generator,
            );

            (
                large_lwe_secret_key,
                glwe_secret_key,
                small_lwe_secret_key,
                small_lwe_zero_encryptions,
                fourier_bsk,
                ksk,
                auto_keys,
                scheme_switch_key,
            )
        };

        Self {
            params,
            large_lwe_secret_key,
            glwe_secret_key,
            small_lwe_secret_key,
            small_lwe_zero_encryptions,
            fourier_bsk,
            ksk_large_to_small: ksk,
            auto_keys,
            scheme_switch_key,
            deterministic_seed,
            next_seed: AtomicU64::new(0),
        }
    }

    fn new_with_shared_glwe_secret_internal(
        params: IntLheParam<u64>,
        glwe_secret_key: &GlweSecretKey<Vec<u64>>,
        deterministic_seed: Option<u128>,
    ) -> Self {
        assert_eq!(
            glwe_secret_key.glwe_dimension(),
            params.glwe_dimension(),
            "shared GLWE secret dimension must match the parameter set"
        );
        assert_eq!(
            glwe_secret_key.polynomial_size(),
            params.polynomial_size(),
            "shared GLWE secret polynomial size must match the parameter set"
        );

        let expected_large_lwe_secret_key = glwe_secret_key.clone().into_lwe_secret_key();

        let (
            small_lwe_secret_key,
            small_lwe_zero_encryptions,
            fourier_bsk,
            ksk_large_to_small,
            auto_keys,
            scheme_switch_key,
        ) = if let Some(seed) = deterministic_seed {
            let mut seeder = DeterministicSeeder::<ActivatedRandomGenerator>::new(Seed(seed));
            let mut secret_generator =
                SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());
            let mut encryption_generator =
                EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(
                    seeder.seed(),
                    &mut seeder,
                );

            let small_lwe_secret_key =
                LweSecretKey::generate_new_binary(params.lwe_dimension(), &mut secret_generator);

            let bootstrap_key = allocate_and_generate_new_lwe_bootstrap_key(
                &small_lwe_secret_key,
                glwe_secret_key,
                params.pbs_base_log(),
                params.pbs_level(),
                params.glwe_modular_std_dev(),
                params.ciphertext_modulus(),
                &mut encryption_generator,
            );

            let mut fourier_bsk = FourierLweBootstrapKey::new(
                bootstrap_key.input_lwe_dimension(),
                bootstrap_key.glwe_size(),
                bootstrap_key.polynomial_size(),
                bootstrap_key.decomposition_base_log(),
                bootstrap_key.decomposition_level_count(),
            );
            convert_standard_lwe_bootstrap_key_to_fourier(&bootstrap_key, &mut fourier_bsk);

            let ksk_large_to_small = allocate_and_generate_new_lwe_keyswitch_key(
                &expected_large_lwe_secret_key,
                &small_lwe_secret_key,
                params.ks_base_log(),
                params.ks_level(),
                params.lwe_modular_std_dev(),
                params.ciphertext_modulus(),
                &mut encryption_generator,
            );

            let auto_keys = gen_all_auto_keys(
                params.auto_base_log(),
                params.auto_level(),
                params.fft_type_auto(),
                glwe_secret_key,
                params.glwe_modular_std_dev(),
                &mut encryption_generator,
            );

            let scheme_switch_key = generate_scheme_switching_key(
                glwe_secret_key,
                params.ss_base_log(),
                params.ss_level(),
                params.glwe_modular_std_dev(),
                params.ciphertext_modulus(),
                &mut encryption_generator,
            );

            let small_lwe_zero_encryptions = generate_small_lwe_zero_encryptions(
                &params,
                &small_lwe_secret_key,
                &mut encryption_generator,
            );

            (
                small_lwe_secret_key,
                small_lwe_zero_encryptions,
                fourier_bsk,
                ksk_large_to_small,
                auto_keys,
                scheme_switch_key,
            )
        } else {
            let mut boxed_seeder = new_seeder();
            let seeder = boxed_seeder.as_mut();
            let mut secret_generator =
                SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());
            let mut encryption_generator =
                EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed(), seeder);

            let small_lwe_secret_key =
                LweSecretKey::generate_new_binary(params.lwe_dimension(), &mut secret_generator);

            let bootstrap_key = allocate_and_generate_new_lwe_bootstrap_key(
                &small_lwe_secret_key,
                glwe_secret_key,
                params.pbs_base_log(),
                params.pbs_level(),
                params.glwe_modular_std_dev(),
                params.ciphertext_modulus(),
                &mut encryption_generator,
            );

            let mut fourier_bsk = FourierLweBootstrapKey::new(
                bootstrap_key.input_lwe_dimension(),
                bootstrap_key.glwe_size(),
                bootstrap_key.polynomial_size(),
                bootstrap_key.decomposition_base_log(),
                bootstrap_key.decomposition_level_count(),
            );
            convert_standard_lwe_bootstrap_key_to_fourier(&bootstrap_key, &mut fourier_bsk);

            let ksk_large_to_small = allocate_and_generate_new_lwe_keyswitch_key(
                &expected_large_lwe_secret_key,
                &small_lwe_secret_key,
                params.ks_base_log(),
                params.ks_level(),
                params.lwe_modular_std_dev(),
                params.ciphertext_modulus(),
                &mut encryption_generator,
            );

            let auto_keys = gen_all_auto_keys(
                params.auto_base_log(),
                params.auto_level(),
                params.fft_type_auto(),
                glwe_secret_key,
                params.glwe_modular_std_dev(),
                &mut encryption_generator,
            );

            let scheme_switch_key = generate_scheme_switching_key(
                glwe_secret_key,
                params.ss_base_log(),
                params.ss_level(),
                params.glwe_modular_std_dev(),
                params.ciphertext_modulus(),
                &mut encryption_generator,
            );

            let small_lwe_zero_encryptions = generate_small_lwe_zero_encryptions(
                &params,
                &small_lwe_secret_key,
                &mut encryption_generator,
            );

            (
                small_lwe_secret_key,
                small_lwe_zero_encryptions,
                fourier_bsk,
                ksk_large_to_small,
                auto_keys,
                scheme_switch_key,
            )
        };

        Self {
            params,
            large_lwe_secret_key: expected_large_lwe_secret_key,
            glwe_secret_key: glwe_secret_key.clone(),
            small_lwe_secret_key,
            small_lwe_zero_encryptions,
            fourier_bsk,
            ksk_large_to_small,
            auto_keys,
            scheme_switch_key,
            deterministic_seed,
            next_seed: AtomicU64::new(0),
        }
    }

    pub fn encrypt_boolean_input(&self, bit: bool) -> LweCiphertext<Vec<u64>> {
        self.encrypt_large_lwe_encoded((bit as u64) << 63)
    }

    pub fn encrypt_small_lwe_encoded(&self, torus_message: u64) -> LweCiphertext<Vec<u64>> {
        let mut encryption_generator = self.new_encryption_generator(0x10);

        allocate_and_encrypt_new_lwe_ciphertext(
            &self.small_lwe_secret_key,
            Plaintext(torus_message),
            self.params.lwe_modular_std_dev(),
            self.params.ciphertext_modulus(),
            &mut encryption_generator,
        )
    }

    pub fn encrypt_large_lwe_encoded(&self, torus_message: u64) -> LweCiphertext<Vec<u64>> {
        let mut encryption_generator = self.new_encryption_generator(0x20);

        allocate_and_encrypt_new_lwe_ciphertext(
            &self.large_lwe_secret_key,
            Plaintext(torus_message),
            self.params.glwe_modular_std_dev(),
            self.params.ciphertext_modulus(),
            &mut encryption_generator,
        )
    }

    pub fn bootstrap_selector(&self, lwe_in: &LweCiphertext<Vec<u64>>) -> SelectorCiphertext {
        assert_eq!(
            lwe_in.lwe_size(),
            self.large_lwe_secret_key.lwe_dimension().to_lwe_size(),
            "selector bootstrap expects a large-LWE input"
        );

        let mut small_lwe = self.keyswitch_large_to_small(lwe_in);
        self.improve_small_lwe_for_blind_rotate(&mut small_lwe, self.params.log_lut_count());
        circuit_bootstrap_lwe_ciphertext_by_trace_with_preprocessing(
            small_lwe.as_view(),
            self.fourier_bsk.as_view(),
            &self.auto_keys,
            self.scheme_switch_key.as_view(),
            self.params.cbs_base_log(),
            self.params.cbs_level(),
            self.params.log_lut_count(),
        )
    }

    pub fn bootstrap_selectors_multi(
        &self,
        lwe_inputs: &[LweCiphertext<Vec<u64>>],
    ) -> Vec<SelectorCiphertext> {
        lwe_inputs
            .iter()
            .map(|lwe_in| self.bootstrap_selector(lwe_in))
            .collect()
    }

    pub fn bootstrap_boolean_input(&self, bit: bool) -> SelectorCiphertext {
        let lwe = self.encrypt_boolean_input(bit);
        self.bootstrap_selector(&lwe)
    }

    pub fn encrypt_packed_boolean_group_input(
        &self,
        bits: [bool; FUSED_SELECTOR_GROUP_BITS],
    ) -> LweCiphertext<Vec<u64>> {
        self.encrypt_large_lwe_encoded(encode_fused_selector_group(bits))
    }

    pub fn encrypt_standard_br_boolean_group_input(
        &self,
        bits: [bool; FUSED_SELECTOR_GROUP_BITS],
    ) -> LweCiphertext<Vec<u64>> {
        self.encrypt_large_lwe_encoded(apply_standard_br_input_bias(
            encode_standard_br_boolean_group(bits),
        ))
    }

    pub fn debug_standard_br_group_pbs_input(
        &self,
        bits: [bool; FUSED_SELECTOR_GROUP_BITS],
    ) -> LweCiphertext<Vec<u64>> {
        let packed_lwe = self.encrypt_standard_br_boolean_group_input(bits);
        self.debug_standard_br_group_pbs_input_from_large_lwe(&packed_lwe)
    }

    pub fn debug_standard_br_group_pbs_input_from_large_lwe(
        &self,
        packed_lwe: &LweCiphertext<Vec<u64>>,
    ) -> LweCiphertext<Vec<u64>> {
        let mut small_lwe = self.keyswitch_large_to_small(&packed_lwe);
        apply_standard_br_input_bias_to_lwe(&mut small_lwe);
        small_lwe
    }

    pub fn bootstrap_selector_group_from_packed_large_lwe(
        &self,
        packed_lwe: &LweCiphertext<Vec<u64>>,
    ) -> Vec<SelectorCiphertext> {
        assert_eq!(
            packed_lwe.lwe_size(),
            self.large_lwe_secret_key.lwe_dimension().to_lwe_size(),
            "fused selector bootstrap expects a large-LWE input"
        );
        let small_lwe = self.keyswitch_large_to_small(packed_lwe);
        self.bootstrap_selector_group_from_packed_small_lwe(&small_lwe)
    }

    pub fn bootstrap_selector_group_standard_br_from_large_lwe(
        &self,
        packed_lwe: &LweCiphertext<Vec<u64>>,
    ) -> Vec<SelectorCiphertext> {
        assert_eq!(
            packed_lwe.lwe_size(),
            self.large_lwe_secret_key.lwe_dimension().to_lwe_size(),
            "standard-br selector bootstrap expects a large-LWE input"
        );
        let mut small_lwe = self.keyswitch_large_to_small(packed_lwe);
        apply_standard_br_input_bias_to_lwe(&mut small_lwe);
        self.bootstrap_selector_group_standard_br_from_small_lwe(&small_lwe)
    }

    pub fn bootstrap_selector_group_standard_br_spread_from_large_lwe(
        &self,
        packed_lwe: &LweCiphertext<Vec<u64>>,
    ) -> Vec<SelectorCiphertext> {
        self.bootstrap_selector_group_standard_br_spread_from_large_lwe_partial(
            packed_lwe,
            FUSED_SELECTOR_GROUP_BITS,
        )
    }

    fn bootstrap_selector_group_standard_br_spread_from_large_lwe_partial(
        &self,
        packed_lwe: &LweCiphertext<Vec<u64>>,
        used_selector_count: usize,
    ) -> Vec<SelectorCiphertext> {
        assert_eq!(
            packed_lwe.lwe_size(),
            self.large_lwe_secret_key.lwe_dimension().to_lwe_size(),
            "standard-br spread selector bootstrap expects a large-LWE input"
        );
        let mut small_lwe = self.keyswitch_large_to_small(packed_lwe);
        apply_standard_br_input_bias_to_lwe(&mut small_lwe);
        self.bootstrap_selector_group_standard_br_from_small_lwe_with_layout(
            &small_lwe,
            INTER_GROUP_STANDARD_BR_SLOT_START,
            INTER_GROUP_STANDARD_BR_SLOT_STRIDE,
            used_selector_count,
        )
    }

    pub fn bootstrap_selector_group_standard_br_from_small_lwe(
        &self,
        packed_lwe: &LweCiphertext<Vec<u64>>,
    ) -> Vec<SelectorCiphertext> {
        self.bootstrap_selector_group_standard_br_from_small_lwe_via_new_pbs(packed_lwe)
    }

    fn bootstrap_selector_group_standard_br_from_small_lwe_with_layout(
        &self,
        packed_lwe: &LweCiphertext<Vec<u64>>,
        slot_start: usize,
        slot_stride: usize,
        used_selector_count: usize,
    ) -> Vec<SelectorCiphertext> {
        assert_eq!(
            packed_lwe.lwe_size(),
            self.small_lwe_secret_key.lwe_dimension().to_lwe_size(),
            "standard-br selector bootstrap expects a small-LWE input"
        );
        assert!(
            (1..=FUSED_SELECTOR_GROUP_BITS).contains(&used_selector_count),
            "used selector count must be in 1..={}",
            FUSED_SELECTOR_GROUP_BITS,
        );
        assert_eq!(
            slot_start + slot_stride * ((1 << FUSED_SELECTOR_GROUP_BITS) - 1),
            slot_start + slot_stride * ((1 << FUSED_SELECTOR_GROUP_BITS) - 1),
        );
        assert!(
            slot_start + slot_stride * ((1 << FUSED_SELECTOR_GROUP_BITS) - 1)
                < FUSED_SELECTOR_LUT_COUNT,
            "standard-br layout exceeds available slot count"
        );

        let glwe_size = self.params.glwe_dimension().to_glwe_size();
        let polynomial_size = self.params.polynomial_size();
        let ciphertext_modulus = self.params.ciphertext_modulus();
        let cbs_base_log = self.params.cbs_base_log();
        let cbs_level = self.params.cbs_level();
        let used_outputs = used_selector_count * cbs_level.0;
        let bootstrapped_levels = self
            .shared_manylut_selector_bootstrap_levels(
                packed_lwe,
                FUSED_SELECTOR_LUT_COUNT,
                FUSED_SELECTOR_GROUP_BITS,
                used_outputs,
                slot_start,
                slot_stride,
            )
            .expect("shared manylut selector bootstrap should fit the configured layout");

        let mut selectors = Vec::with_capacity(used_selector_count);
        for selector_idx in 0..used_selector_count {
            let mut glev = GlweCiphertextList::new(
                0u64,
                glwe_size,
                polynomial_size,
                GlweCiphertextCount(cbs_level.0),
                ciphertext_modulus,
            );

            for level_idx in 0..cbs_level.0 {
                let output_idx = selector_idx * cbs_level.0 + level_idx;
                let mut preprocessed = LweCiphertext::new(
                    0u64,
                    self.large_lwe_secret_key.lwe_dimension().to_lwe_size(),
                    ciphertext_modulus,
                );
                extract_lwe_sample_from_glwe_ciphertext(
                    &bootstrapped_levels[output_idx],
                    &mut preprocessed,
                    MonomialDegree(0),
                );
                let log_scale = u64::BITS as usize - (level_idx + 1) * cbs_base_log.0;
                lwe_ciphertext_plaintext_add_assign(
                    &mut preprocessed,
                    Plaintext(1u64 << (log_scale - 1)),
                );
                lwe_preprocessing_assign(&mut preprocessed, polynomial_size);

                let mut glwe = glev.get_mut(level_idx);
                convert_lwe_to_glwe_const(&preprocessed, &mut glwe);
                trace_assign(&mut glwe, &self.auto_keys);
            }

            let mut ggsw = GgswCiphertext::new(
                0u64,
                glwe_size,
                polynomial_size,
                cbs_base_log,
                cbs_level,
                ciphertext_modulus,
            );
            switch_scheme(&glev, &mut ggsw, self.scheme_switch_key.as_view());

            let mut fourier =
                FourierGgswCiphertext::new(glwe_size, polynomial_size, cbs_base_log, cbs_level);
            convert_standard_ggsw_ciphertext_to_fourier(&ggsw, &mut fourier);
            selectors.push(clone_fourier_ggsw(fourier.as_view()));
        }

        selectors
    }

    fn bootstrap_selector_group_standard_br_from_small_lwe_via_new_pbs(
        &self,
        packed_lwe: &LweCiphertext<Vec<u64>>,
    ) -> Vec<SelectorCiphertext> {
        assert_eq!(
            packed_lwe.lwe_size(),
            self.small_lwe_secret_key.lwe_dimension().to_lwe_size(),
            "standard-br selector bootstrap expects a small-LWE input"
        );

        let glwe_size = self.params.glwe_dimension().to_glwe_size();
        let polynomial_size = self.params.polynomial_size();
        let ciphertext_modulus = self.params.ciphertext_modulus();
        let cbs_base_log = self.params.cbs_base_log();
        let cbs_level = self.params.cbs_level();
        let used_outputs = FUSED_SELECTOR_GROUP_BITS * cbs_level.0;
        let bootstrapped_levels = self
            .shared_manylut_selector_bootstrap_levels(
                packed_lwe,
                FUSED_SELECTOR_LUT_COUNT,
                FUSED_SELECTOR_GROUP_BITS,
                used_outputs,
                DENSE_STANDARD_BR_SLOT_START,
                1,
            )
            .expect("shared manylut selector bootstrap should fit the configured layout");

        let mut selectors = Vec::with_capacity(FUSED_SELECTOR_GROUP_BITS);
        for selector_idx in 0..FUSED_SELECTOR_GROUP_BITS {
            let mut glev = GlweCiphertextList::new(
                0u64,
                glwe_size,
                polynomial_size,
                GlweCiphertextCount(cbs_level.0),
                ciphertext_modulus,
            );

            for level_idx in 0..cbs_level.0 {
                let output_idx = selector_idx * cbs_level.0 + level_idx;
                let mut preprocessed = LweCiphertext::new(
                    0u64,
                    self.large_lwe_secret_key.lwe_dimension().to_lwe_size(),
                    ciphertext_modulus,
                );
                extract_lwe_sample_from_glwe_ciphertext(
                    &bootstrapped_levels[output_idx],
                    &mut preprocessed,
                    MonomialDegree(0),
                );
                let log_scale = u64::BITS as usize - (level_idx + 1) * cbs_base_log.0;
                lwe_ciphertext_plaintext_add_assign(
                    &mut preprocessed,
                    Plaintext(1u64 << (log_scale - 1)),
                );
                lwe_preprocessing_assign(&mut preprocessed, polynomial_size);

                let mut glwe = glev.get_mut(level_idx);
                convert_lwe_to_glwe_const(&preprocessed, &mut glwe);
                trace_assign(&mut glwe, &self.auto_keys);
            }

            let mut ggsw = GgswCiphertext::new(
                0u64,
                glwe_size,
                polynomial_size,
                cbs_base_log,
                cbs_level,
                ciphertext_modulus,
            );
            switch_scheme(&glev, &mut ggsw, self.scheme_switch_key.as_view());

            let mut fourier =
                FourierGgswCiphertext::new(glwe_size, polynomial_size, cbs_base_log, cbs_level);
            convert_standard_ggsw_ciphertext_to_fourier(&ggsw, &mut fourier);
            selectors.push(clone_fourier_ggsw(fourier.as_view()));
        }

        selectors
    }

    fn shared_manylut_selector_bootstrap_levels(
        &self,
        lwe: &LweCiphertext<Vec<u64>>,
        _slot_count: usize,
        _slot_log: usize,
        _used_outputs: usize,
        _slot_start: usize,
        _slot_stride: usize,
    ) -> Result<Vec<GlweCiphertext<Vec<u64>>>, &'static str> {
        Ok(self.shared_manylut_selector_bootstrap_levels_via_passed_pbs(lwe))
    }

    fn shared_manylut_selector_bootstrap_levels_via_passed_pbs(
        &self,
        small_lwe: &LweCiphertext<Vec<u64>>,
    ) -> Vec<GlweCiphertext<Vec<u64>>> {
        const SLOT_COUNT: usize = 8;
        const MANYLUT_LOG_LUT_COUNT: usize = 1;
        let glwe_size = self.params.glwe_dimension().to_glwe_size();
        let polynomial_size = self.params.polynomial_size();
        let ciphertext_modulus = self.params.ciphertext_modulus();
        let padding = polynomial_size.0 / SLOT_COUNT;
        let cbs_base_log = self.params.cbs_base_log();
        let cbs_level = self.params.cbs_level();
        let lut_count = cbs_level.0;

        let mut accumulator = vec![0u64; polynomial_size.0];
        for i in 0..padding {
            let k = i % lut_count;
            let log_scale = u64::BITS as usize - (k + 1) * cbs_base_log.0;
            accumulator[i] = 1u64 << (log_scale - 1);
        }

        let accumulator_plaintext = PlaintextList::from_container(accumulator);
        let accumulator = allocate_and_trivially_encrypt_new_glwe_ciphertext(
            glwe_size,
            &accumulator_plaintext,
            ciphertext_modulus,
        );

        let mut buffers = ComputationBuffers::new();
        let fft = Fft::new(polynomial_size);
        let fft_view = fft.as_view();
        buffers.resize(
            programmable_bootstrap_lwe_ciphertext_mem_optimized_requirement::<u64>(
                glwe_size,
                polynomial_size,
                fft_view,
            )
            .unwrap()
            .unaligned_bytes_required(),
        );
        let stack = buffers.stack();

        let (mut local_accumulator_data, stack) = stack.collect_aligned(
            aligned_vec::CACHELINE_ALIGN,
            accumulator.as_ref().iter().copied(),
        );
        let mut local_accumulator = GlweCiphertextMutView::from_container(
            &mut *local_accumulator_data,
            polynomial_size,
            ciphertext_modulus,
        );
        let mut improved_small_lwe = small_lwe.clone();
        self.improve_small_lwe_for_blind_rotate(
            &mut improved_small_lwe,
            LutCountLog(MANYLUT_LOG_LUT_COUNT),
        );

        gen_blind_rotate_local_assign(
            self.fourier_bsk.as_view(),
            local_accumulator.as_mut_view(),
            ModulusSwitchOffset(0),
            LutCountLog(MANYLUT_LOG_LUT_COUNT),
            improved_small_lwe.as_ref(),
            fft_view,
            stack,
        );

        let acc = GlweCiphertext::from_container(
            local_accumulator.as_ref().to_vec(),
            polynomial_size,
            ciphertext_modulus,
        );

        let mut extracted = Vec::with_capacity(cbs_level.0);
        for k in 0..cbs_level.0 {
            let mut buf = GlweCiphertext::new(
                0u64,
                acc.glwe_size(),
                acc.polynomial_size(),
                acc.ciphertext_modulus(),
            );
            glwe_ciphertext_clone_from(&mut buf, &acc);
            glwe_ciphertext_monic_monomial_div_assign(&mut buf, MonomialDegree(k));
            extracted.push(buf);
        }

        let mut outputs = Vec::with_capacity(FUSED_SELECTOR_GROUP_BITS * cbs_level.0);
        for selector_idx in 0..FUSED_SELECTOR_GROUP_BITS {
            let poly = selector_truth_polynomial_manylut8(polynomial_size, selector_idx);
            let masked0 = multiply_glwe_by_sparse_clear_polynomial(&extracted[0], &poly);
            let masked1 = multiply_glwe_by_sparse_clear_polynomial(&extracted[1], &poly);
            outputs.push(masked0);
            outputs.push(masked1);
        }
        outputs
    }

    pub fn bootstrap_selector_group_from_packed_small_lwe(
        &self,
        packed_lwe: &LweCiphertext<Vec<u64>>,
    ) -> Vec<SelectorCiphertext> {
        assert_eq!(
            packed_lwe.lwe_size(),
            self.small_lwe_secret_key.lwe_dimension().to_lwe_size(),
            "fused selector bootstrap expects a small-LWE input"
        );

        let common_carrier = self.shared_pbs_multi_common_carrier(packed_lwe);
        let glwe_size = self.params.glwe_dimension().to_glwe_size();
        let polynomial_size = self.params.polynomial_size();
        let ciphertext_modulus = self.params.ciphertext_modulus();
        let cbs_base_log = self.params.cbs_base_log();
        let cbs_level = self.params.cbs_level();
        let padding = polynomial_size.0 / FUSED_SELECTOR_LUT_COUNT;
        let carrier_start_slot = FUSED_SELECTOR_LUT_COUNT - cbs_level.0;

        let mut selectors = Vec::with_capacity(FUSED_SELECTOR_GROUP_BITS);
        for selector_idx in 0..FUSED_SELECTOR_GROUP_BITS {
            let mut glev = GlweCiphertextList::new(
                0u64,
                glwe_size,
                polynomial_size,
                GlweCiphertextCount(cbs_level.0),
                ciphertext_modulus,
            );

            for level_idx in 0..cbs_level.0 {
                let mut aligned_glwe =
                    GlweCiphertext::new(0u64, glwe_size, polynomial_size, ciphertext_modulus);
                glwe_ciphertext_clone_from(&mut aligned_glwe, &common_carrier);
                glwe_ciphertext_monic_monomial_div_assign(
                    &mut aligned_glwe,
                    MonomialDegree((carrier_start_slot + level_idx) * padding),
                );

                let masked_glwe = multiply_glwe_by_clear_polynomial(
                    &aligned_glwe,
                    &selector_truth_polynomial(polynomial_size, selector_idx),
                );

                let mut shifted_glwe = masked_glwe;

                let log_scale = u64::BITS as usize - (level_idx + 1) * cbs_base_log.0;
                glwe_ciphertext_plaintext_add_assign(
                    &mut shifted_glwe,
                    Plaintext(1u64 << (log_scale - 1)),
                );

                let mut preprocessed = LweCiphertext::new(
                    0u64,
                    self.large_lwe_secret_key.lwe_dimension().to_lwe_size(),
                    ciphertext_modulus,
                );
                extract_lwe_sample_from_glwe_ciphertext(
                    &shifted_glwe,
                    &mut preprocessed,
                    MonomialDegree(0),
                );
                lwe_preprocessing_assign(&mut preprocessed, polynomial_size);

                let mut glwe = glev.get_mut(level_idx);
                convert_lwe_to_glwe_const(&preprocessed, &mut glwe);
                trace_assign(&mut glwe, &self.auto_keys);
            }

            let mut ggsw = GgswCiphertext::new(
                0u64,
                glwe_size,
                polynomial_size,
                cbs_base_log,
                cbs_level,
                ciphertext_modulus,
            );
            switch_scheme(&glev, &mut ggsw, self.scheme_switch_key.as_view());

            let mut fourier =
                FourierGgswCiphertext::new(glwe_size, polynomial_size, cbs_base_log, cbs_level);
            convert_standard_ggsw_ciphertext_to_fourier(&ggsw, &mut fourier);
            selectors.push(clone_fourier_ggsw(fourier.as_view()));
        }

        selectors
    }

    pub fn bootstrap_boolean_group_fused(
        &self,
        bits: [bool; FUSED_SELECTOR_GROUP_BITS],
    ) -> Vec<SelectorCiphertext> {
        let packed_lwe = self.encrypt_packed_boolean_group_input(bits);
        self.bootstrap_selector_group_from_packed_large_lwe(&packed_lwe)
    }

    pub fn bootstrap_boolean_group_standard_br(
        &self,
        bits: [bool; FUSED_SELECTOR_GROUP_BITS],
    ) -> Vec<SelectorCiphertext> {
        let packed_lwe = self.encrypt_standard_br_boolean_group_input(bits);
        self.bootstrap_selector_group_standard_br_from_large_lwe(&packed_lwe)
    }

    pub fn debug_bootstrap_boolean_group_standard_br_pretrace(
        &self,
        bits: [bool; FUSED_SELECTOR_GROUP_BITS],
    ) -> Vec<i128> {
        let small_lwe = self.debug_standard_br_group_pbs_input(bits);
        let slot_start = DENSE_STANDARD_BR_SLOT_START;
        let slot_stride = 1;
        let cbs_base_log = self.params.cbs_base_log();
        let cbs_level = self.params.cbs_level();
        let used_outputs = FUSED_SELECTOR_GROUP_BITS * cbs_level.0;
        let bootstrapped_levels = self
            .shared_manylut_selector_bootstrap_levels(
                &small_lwe,
                FUSED_SELECTOR_LUT_COUNT,
                FUSED_SELECTOR_GROUP_BITS,
                used_outputs,
                slot_start,
                slot_stride,
            )
            .expect("shared manylut selector bootstrap should fit the configured layout");

        let mut out = Vec::with_capacity(used_outputs);
        for selector_idx in 0..FUSED_SELECTOR_GROUP_BITS {
            for level_idx in 0..cbs_level.0 {
                let output_idx = selector_idx * cbs_level.0 + level_idx;
                let mut lwe = self.extract_coefficient0_as_lwe(&bootstrapped_levels[output_idx]);
                let log_scale = u64::BITS as usize - (level_idx + 1) * cbs_base_log.0;
                lwe_ciphertext_plaintext_add_assign(&mut lwe, Plaintext(1u64 << (log_scale - 1)));
                out.push(center_u64_local(lwe_phase_unsigned_local(
                    &lwe,
                    &self.large_lwe_secret_key,
                )));
            }
        }
        out
    }

    pub fn debug_bootstrap_boolean_group_standard_br_slots(
        &self,
        bits: [bool; FUSED_SELECTOR_GROUP_BITS],
    ) -> Vec<i128> {
        let small_lwe = self.debug_standard_br_group_pbs_input(bits);
        let glwe_size = self.params.glwe_dimension().to_glwe_size();
        let polynomial_size = self.params.polynomial_size();
        let ciphertext_modulus = self.params.ciphertext_modulus();
        let used_outputs = FUSED_SELECTOR_GROUP_BITS * self.params.cbs_level().0;

        let bootstrapped_levels = self
            .shared_manylut_selector_bootstrap_levels(
                &small_lwe,
                FUSED_SELECTOR_LUT_COUNT,
                FUSED_SELECTOR_GROUP_BITS,
                used_outputs,
                DENSE_STANDARD_BR_SLOT_START,
                1,
            )
            .expect("shared manylut selector bootstrap should fit the configured layout");

        let mut pt = PlaintextList::new(0u64, PlaintextCount(polynomial_size.0));
        let mut values = Vec::new();
        let padding = polynomial_size.0 / (1usize << FUSED_SELECTOR_GROUP_BITS);

        let mut glwe = GlweCiphertext::new(0u64, glwe_size, polynomial_size, ciphertext_modulus);
        glwe_ciphertext_clone_from(&mut glwe, &bootstrapped_levels[0]);
        decrypt_glwe_ciphertext(&self.glwe_secret_key, &glwe, &mut pt);
        for slot in 0..(1usize << FUSED_SELECTOR_GROUP_BITS) {
            values.push(center_u64_local(pt.as_ref()[slot * padding]));
        }
        values
    }

    pub fn debug_bootstrap_boolean_group_standard_br_trace_levels(
        &self,
        bits: [bool; FUSED_SELECTOR_GROUP_BITS],
    ) -> Vec<(usize, usize, i128, i128)> {
        let small_lwe = self.debug_standard_br_group_pbs_input(bits);
        let slot_start = DENSE_STANDARD_BR_SLOT_START;
        let slot_stride = 1;
        let glwe_size = self.params.glwe_dimension().to_glwe_size();
        let polynomial_size = self.params.polynomial_size();
        let ciphertext_modulus = self.params.ciphertext_modulus();
        let cbs_base_log = self.params.cbs_base_log();
        let cbs_level = self.params.cbs_level();
        let used_outputs = FUSED_SELECTOR_GROUP_BITS * cbs_level.0;
        let bootstrapped_levels = self
            .shared_manylut_selector_bootstrap_levels(
                &small_lwe,
                FUSED_SELECTOR_LUT_COUNT,
                FUSED_SELECTOR_GROUP_BITS,
                used_outputs,
                slot_start,
                slot_stride,
            )
            .expect("shared manylut selector bootstrap should fit the configured layout");

        let mut out = Vec::with_capacity(used_outputs);
        for selector_idx in 0..FUSED_SELECTOR_GROUP_BITS {
            for level_idx in 0..cbs_level.0 {
                let output_idx = selector_idx * cbs_level.0 + level_idx;
                let mut preprocessed = LweCiphertext::new(
                    0u64,
                    self.large_lwe_secret_key.lwe_dimension().to_lwe_size(),
                    ciphertext_modulus,
                );
                extract_lwe_sample_from_glwe_ciphertext(
                    &bootstrapped_levels[output_idx],
                    &mut preprocessed,
                    MonomialDegree(0),
                );
                let log_scale = u64::BITS as usize - (level_idx + 1) * cbs_base_log.0;
                lwe_ciphertext_plaintext_add_assign(
                    &mut preprocessed,
                    Plaintext(1u64 << (log_scale - 1)),
                );
                let pretrace = center_u64_local(lwe_phase_unsigned_local(
                    &preprocessed,
                    &self.large_lwe_secret_key,
                ));

                lwe_preprocessing_assign(&mut preprocessed, polynomial_size);
                let mut glwe =
                    GlweCiphertext::new(0u64, glwe_size, polynomial_size, ciphertext_modulus);
                convert_lwe_to_glwe_const(&preprocessed, &mut glwe);
                trace_assign(&mut glwe, &self.auto_keys);
                let traced_lwe = self.extract_coefficient0_as_lwe(&glwe);
                let posttrace = center_u64_local(lwe_phase_unsigned_local(
                    &traced_lwe,
                    &self.large_lwe_secret_key,
                ));

                out.push((selector_idx, level_idx, pretrace, posttrace));
            }
        }
        out
    }

    pub fn debug_bootstrap_boolean_group_standard_br_raw_levels(
        &self,
        bits: [bool; FUSED_SELECTOR_GROUP_BITS],
    ) -> Vec<i128> {
        let small_lwe = self.debug_standard_br_group_pbs_input(bits);
        let bootstrapped_levels =
            self.shared_manylut_selector_bootstrap_levels_via_passed_pbs(&small_lwe);

        let mut out = Vec::with_capacity(FUSED_SELECTOR_GROUP_BITS * self.params.cbs_level().0);
        for selector_idx in 0..FUSED_SELECTOR_GROUP_BITS {
            for level_idx in 0..self.params.cbs_level().0 {
                let output_idx = selector_idx * self.params.cbs_level().0 + level_idx;
                let lwe = self.extract_coefficient0_as_lwe(&bootstrapped_levels[output_idx]);
                out.push(center_u64_local(lwe_phase_unsigned_local(
                    &lwe,
                    &self.large_lwe_secret_key,
                )));
            }
        }
        out
    }

    pub fn debug_bootstrap_boolean_group_standard_br_rounded_levels(
        &self,
        bits: [bool; FUSED_SELECTOR_GROUP_BITS],
    ) -> Vec<i128> {
        let raw = self.debug_bootstrap_boolean_group_standard_br_raw_levels(bits);
        let mut out = Vec::with_capacity(raw.len());
        for selector_idx in 0..FUSED_SELECTOR_GROUP_BITS {
            for level_idx in 0..self.params.cbs_level().0 {
                let idx = selector_idx * self.params.cbs_level().0 + level_idx;
                let log_scale = u64::BITS as usize - (level_idx + 1) * self.params.cbs_base_log().0;
                out.push(raw[idx] + (1i128 << (log_scale - 1)));
            }
        }
        out
    }

    pub fn debug_bootstrap_boolean_group_standard_br_single_selector_tail(
        &self,
        bits: [bool; FUSED_SELECTOR_GROUP_BITS],
        selector_idx: usize,
        delta: u64,
    ) -> (Vec<i128>, Vec<i128>, Vec<i128>, u64) {
        let packed_lwe = self.encrypt_standard_br_boolean_group_input(bits);
        self.debug_bootstrap_boolean_group_standard_br_single_selector_tail_from_large_lwe(
            &packed_lwe,
            selector_idx,
            delta,
        )
    }

    pub fn debug_bootstrap_boolean_group_standard_br_single_selector_tail_from_large_lwe(
        &self,
        packed_lwe: &LweCiphertext<Vec<u64>>,
        selector_idx: usize,
        delta: u64,
    ) -> (Vec<i128>, Vec<i128>, Vec<i128>, u64) {
        assert!(
            selector_idx < FUSED_SELECTOR_GROUP_BITS,
            "selector_idx must be smaller than the grouped selector width"
        );

        let small_lwe = self.debug_standard_br_group_pbs_input_from_large_lwe(packed_lwe);
        let bootstrapped_levels =
            self.shared_manylut_selector_bootstrap_levels_via_passed_pbs(&small_lwe);

        let glwe_size = self.params.glwe_dimension().to_glwe_size();
        let polynomial_size = self.params.polynomial_size();
        let ciphertext_modulus = self.params.ciphertext_modulus();
        let cbs_base_log = self.params.cbs_base_log();
        let cbs_level = self.params.cbs_level();

        let mut raw = Vec::with_capacity(cbs_level.0);
        let mut rounded = Vec::with_capacity(cbs_level.0);
        let mut traced = Vec::with_capacity(cbs_level.0);

        let mut glev = GlweCiphertextList::new(
            0u64,
            glwe_size,
            polynomial_size,
            GlweCiphertextCount(cbs_level.0),
            ciphertext_modulus,
        );

        for level_idx in 0..cbs_level.0 {
            let output_idx = selector_idx * cbs_level.0 + level_idx;

            let mut preprocessed = LweCiphertext::new(
                0u64,
                self.large_lwe_secret_key.lwe_dimension().to_lwe_size(),
                ciphertext_modulus,
            );
            extract_lwe_sample_from_glwe_ciphertext(
                &bootstrapped_levels[output_idx],
                &mut preprocessed,
                MonomialDegree(0),
            );
            raw.push(center_u64_local(lwe_phase_unsigned_local(
                &preprocessed,
                &self.large_lwe_secret_key,
            )));

            let log_scale = u64::BITS as usize - (level_idx + 1) * cbs_base_log.0;
            lwe_ciphertext_plaintext_add_assign(
                &mut preprocessed,
                Plaintext(1u64 << (log_scale - 1)),
            );
            rounded.push(center_u64_local(lwe_phase_unsigned_local(
                &preprocessed,
                &self.large_lwe_secret_key,
            )));

            lwe_preprocessing_assign(&mut preprocessed, polynomial_size);
            let mut glwe = glev.get_mut(level_idx);
            convert_lwe_to_glwe_const(&preprocessed, &mut glwe);
            trace_assign(&mut glwe, &self.auto_keys);

            let mut traced_lwe = LweCiphertext::new(
                0u64,
                self.large_lwe_secret_key.lwe_dimension().to_lwe_size(),
                ciphertext_modulus,
            );
            extract_lwe_sample_from_glwe_ciphertext(&glwe, &mut traced_lwe, MonomialDegree(0));
            traced.push(center_u64_local(lwe_phase_unsigned_local(
                &traced_lwe,
                &self.large_lwe_secret_key,
            )));
        }

        let mut ggsw = GgswCiphertext::new(
            0u64,
            glwe_size,
            polynomial_size,
            cbs_base_log,
            cbs_level,
            ciphertext_modulus,
        );
        switch_scheme(&glev, &mut ggsw, self.scheme_switch_key.as_view());

        let mut selector =
            FourierGgswCiphertext::new(glwe_size, polynomial_size, cbs_base_log, cbs_level);
        convert_standard_ggsw_ciphertext_to_fourier(&ggsw, &mut selector);

        let mut zero = self.encrypt_glwe_constant(0, delta);
        let mut one = self.encrypt_glwe_constant(1, delta);
        cmux_assign(&mut zero, &mut one, &selector);
        let actual = self.decrypt_glwe_coefficient0(&zero, delta);

        (raw, rounded, traced, actual)
    }

    pub fn bootstrap_boolean_inputs_multi(&self, bits: &[bool]) -> Vec<SelectorCiphertext> {
        let lwe_inputs = bits
            .iter()
            .map(|bit| self.encrypt_boolean_input(*bit))
            .collect::<Vec<_>>();
        self.bootstrap_selectors_multi(&lwe_inputs)
    }

    pub fn encrypt_glwe_constant(&self, value: u64, delta: u64) -> GlweCiphertext<Vec<u64>> {
        let mut plaintext = vec![0u64; self.params.polynomial_size().0];
        plaintext[0] = value.wrapping_mul(delta);
        let plaintext = PlaintextList::from_container(plaintext);

        let mut encryption_generator = self.new_encryption_generator(0x30);

        let mut ciphertext = GlweCiphertext::new(
            0u64,
            self.params.glwe_dimension().to_glwe_size(),
            self.params.polynomial_size(),
            self.params.ciphertext_modulus(),
        );

        encrypt_glwe_ciphertext(
            &self.glwe_secret_key,
            &mut ciphertext,
            &plaintext,
            self.params.glwe_modular_std_dev(),
            &mut encryption_generator,
        );

        ciphertext
    }

    pub fn trivially_encrypt_glwe_constant(
        &self,
        value: u64,
        delta: u64,
    ) -> GlweCiphertext<Vec<u64>> {
        let mut plaintext = vec![0u64; self.params.polynomial_size().0];
        plaintext[0] = value.wrapping_mul(delta);
        let plaintext = PlaintextList::from_container(plaintext);
        allocate_and_trivially_encrypt_new_glwe_ciphertext(
            self.params.glwe_dimension().to_glwe_size(),
            &plaintext,
            self.params.ciphertext_modulus(),
        )
    }

    fn new_encryption_generator(
        &self,
        stream_tag: u64,
    ) -> EncryptionRandomGenerator<ActivatedRandomGenerator> {
        if let Some(base_seed) = self.deterministic_seed {
            let counter = self.next_seed.fetch_add(1, Ordering::Relaxed) as u128;
            let seed = Seed(base_seed ^ ((stream_tag as u128) << 64) ^ counter);
            let mut seeder = DeterministicSeeder::<ActivatedRandomGenerator>::new(seed);
            EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed(), &mut seeder)
        } else {
            let mut boxed_seeder = new_seeder();
            let seeder = boxed_seeder.as_mut();
            EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed(), seeder)
        }
    }

    pub fn extract_coefficient0_as_lwe(
        &self,
        glwe_in: &GlweCiphertext<Vec<u64>>,
    ) -> LweCiphertext<Vec<u64>> {
        let mut lwe_out = LweCiphertext::new(
            0u64,
            self.large_lwe_secret_key.lwe_dimension().to_lwe_size(),
            self.params.ciphertext_modulus(),
        );

        extract_lwe_sample_from_glwe_ciphertext(glwe_in, &mut lwe_out, MonomialDegree(0));
        lwe_out
    }

    pub fn keyswitch_large_to_small(
        &self,
        lwe_in: &LweCiphertext<Vec<u64>>,
    ) -> LweCiphertext<Vec<u64>> {
        let mut lwe_out = LweCiphertext::new(
            0u64,
            self.small_lwe_secret_key.lwe_dimension().to_lwe_size(),
            self.params.ciphertext_modulus(),
        );

        keyswitch_lwe_ciphertext(&self.ksk_large_to_small, lwe_in, &mut lwe_out);
        lwe_out
    }

    pub(crate) fn improve_small_lwe_for_blind_rotate(
        &self,
        lwe: &mut LweCiphertext<Vec<u64>>,
        lut_count_log: LutCountLog,
    ) {
        let polynomial_log = self.params.polynomial_size().log2().0;
        assert!(
            lut_count_log.0 <= polynomial_log + 1,
            "invalid lut_count_log {} for polynomial size {}",
            lut_count_log.0,
            self.params.polynomial_size().0,
        );

        let log_modulus = CiphertextModulusLog(polynomial_log + 1 - lut_count_log.0);
        let input_variance_modular = self
            .params
            .lwe_modular_std_dev()
            .get_modular_variance(u64::BITS);

        improve_lwe_ciphertext_modulus_switch_noise_for_binary_key(
            lwe,
            &self.small_lwe_zero_encryptions,
            input_variance_modular,
            log_modulus,
            NoiseEstimationMeasureBound(MS_NOISE_REDUCTION_MEASURE_BOUND),
            RSigmaFactor(MS_NOISE_REDUCTION_R_SIGMA),
        );
    }

    pub fn glwe_boolean_to_selector(
        &self,
        glwe_in: &GlweCiphertext<Vec<u64>>,
    ) -> SelectorCiphertext {
        let large_lwe = self.extract_coefficient0_as_lwe(glwe_in);
        self.bootstrap_selector(&large_lwe)
    }

    pub fn glwe_booleans_to_selectors_multi(
        &self,
        glwe_inputs: &[GlweCiphertext<Vec<u64>>],
    ) -> Vec<SelectorCiphertext> {
        let large_lwes = glwe_inputs
            .iter()
            .map(|glwe_in| self.extract_coefficient0_as_lwe(glwe_in))
            .collect::<Vec<_>>();
        self.bootstrap_selectors_multi(&large_lwes)
    }

    pub fn pack_weighted_glwe_outputs_to_large_lwe(
        &self,
        glwe_inputs: &[GlweCiphertext<Vec<u64>>],
    ) -> LweCiphertext<Vec<u64>> {
        assert!(
            (1..=FUSED_SELECTOR_GROUP_BITS).contains(&glwe_inputs.len()),
            "expected between one and three GLWE outputs for selector bootstrap"
        );

        let mut packed_lwe = LweCiphertext::new(
            0u64,
            self.large_lwe_secret_key.lwe_dimension().to_lwe_size(),
            self.params.ciphertext_modulus(),
        );

        for (bit_index, glwe_in) in glwe_inputs.iter().enumerate() {
            // Keep each CMUX tree on the same 0/1 carrier, then apply the
            // 1/2/4 inter-group weighting only after coefficient extraction.
            let mut large_lwe = self.extract_coefficient0_as_lwe(glwe_in);
            let weight = 1u64 << bit_index;
            lwe_ciphertext_cleartext_mul_assign(&mut large_lwe, Cleartext(weight));
            lwe_ciphertext_add_assign(&mut packed_lwe, &large_lwe);
        }

        packed_lwe
    }

    pub fn pack_standard_br_group_weighted_glwe_outputs_to_large_lwe(
        &self,
        glwe_inputs: &[GlweCiphertext<Vec<u64>>],
    ) -> LweCiphertext<Vec<u64>> {
        let mut packed_lwe = self.pack_weighted_glwe_outputs_to_large_lwe(glwe_inputs);
        let base_offset = inter_group_standard_br_base_offset();
        lwe_ciphertext_plaintext_add_assign(&mut packed_lwe, Plaintext(base_offset));
        packed_lwe
    }

    pub fn weighted_glwe_outputs_to_selectors_fused(
        &self,
        glwe_inputs: &[GlweCiphertext<Vec<u64>>],
    ) -> Vec<SelectorCiphertext> {
        let packed_lwe = self.pack_weighted_glwe_outputs_to_large_lwe(glwe_inputs);
        self.bootstrap_selector_group_from_packed_large_lwe(&packed_lwe)
    }

    pub fn weighted_glwe_outputs_to_selectors_standard_br(
        &self,
        glwe_inputs: &[GlweCiphertext<Vec<u64>>],
    ) -> Vec<SelectorCiphertext> {
        let used_selector_count = glwe_inputs.len();
        let packed_lwe =
            self.pack_standard_br_group_weighted_glwe_outputs_to_large_lwe(glwe_inputs);
        self.bootstrap_selector_group_standard_br_spread_from_large_lwe_partial(
            &packed_lwe,
            used_selector_count,
        )
    }

    pub fn packed_large_lwe_to_selectors_standard_br(
        &self,
        packed_lwe: &LweCiphertext<Vec<u64>>,
        used_selector_count: usize,
    ) -> Vec<SelectorCiphertext> {
        self.bootstrap_selector_group_standard_br_spread_from_large_lwe_partial(
            packed_lwe,
            used_selector_count,
        )
    }

    pub fn packed_glwe_to_selectors_standard_br(
        &self,
        packed_glwe: &GlweCiphertext<Vec<u64>>,
        used_selector_count: usize,
    ) -> Vec<SelectorCiphertext> {
        let packed_lwe = self.extract_coefficient0_as_lwe(packed_glwe);
        self.packed_large_lwe_to_selectors_standard_br(&packed_lwe, used_selector_count)
    }

    pub fn decrypt_large_lwe(&self, lwe_in: &LweCiphertext<Vec<u64>>, delta: u64) -> u64 {
        let plaintext = decrypt_lwe_ciphertext(&self.large_lwe_secret_key, lwe_in);
        decode_torus(plaintext.0, delta)
    }

    pub fn decrypt_glwe_coefficient0(&self, glwe_in: &GlweCiphertext<Vec<u64>>, delta: u64) -> u64 {
        let mut plaintext =
            PlaintextList::from_container(vec![0u64; self.params.polynomial_size().0]);
        decrypt_glwe_ciphertext(&self.glwe_secret_key, glwe_in, &mut plaintext);
        decode_torus(plaintext.as_ref()[0], delta)
    }

    #[allow(dead_code)]
    fn shared_pbs_multi_level_carriers(
        &self,
        lwe: &LweCiphertext<Vec<u64>>,
    ) -> Vec<GlweCiphertext<Vec<u64>>> {
        let common_carrier = self.shared_pbs_multi_common_carrier(lwe);
        let glwe_size = self.params.glwe_dimension().to_glwe_size();
        let polynomial_size = self.params.polynomial_size();
        let ciphertext_modulus = self.params.ciphertext_modulus();
        let cbs_level = self.params.cbs_level().0;
        let padding = polynomial_size.0 / FUSED_SELECTOR_LUT_COUNT;
        let carrier_start_slot = FUSED_SELECTOR_LUT_COUNT - cbs_level;

        let mut carriers = Vec::with_capacity(cbs_level);
        for level_idx in 0..cbs_level {
            let mut aligned_glwe =
                GlweCiphertext::new(0u64, glwe_size, polynomial_size, ciphertext_modulus);
            glwe_ciphertext_clone_from(&mut aligned_glwe, &common_carrier);
            glwe_ciphertext_monic_monomial_div_assign(
                &mut aligned_glwe,
                MonomialDegree((carrier_start_slot + level_idx) * padding),
            );
            carriers.push(aligned_glwe);
        }

        carriers
    }

    fn shared_pbs_multi_common_carrier(
        &self,
        lwe: &LweCiphertext<Vec<u64>>,
    ) -> GlweCiphertext<Vec<u64>> {
        let glwe_size = self.params.glwe_dimension().to_glwe_size();
        let polynomial_size = self.params.polynomial_size();
        let ciphertext_modulus = self.params.ciphertext_modulus();
        let cbs_base_log = self.params.cbs_base_log();
        let cbs_level = self.params.cbs_level().0;
        let padding = polynomial_size.0 / FUSED_SELECTOR_LUT_COUNT;

        assert_eq!(polynomial_size.0 % FUSED_SELECTOR_LUT_COUNT, 0);

        let mut accumulator = vec![0u64; polynomial_size.0];
        let start_slot = FUSED_SELECTOR_LUT_COUNT - cbs_level;
        for level_idx in 0..cbs_level {
            let amplitude = 1u64 << (64 - (level_idx + 1) * cbs_base_log.0 - 1);
            let slot = start_slot + level_idx;
            for coeff in accumulator.iter_mut().skip(slot * padding).take(padding) {
                *coeff = amplitude;
            }
        }

        let accumulator_plaintext = PlaintextList::from_container(accumulator);
        let accumulator = allocate_and_trivially_encrypt_new_glwe_ciphertext(
            glwe_size,
            &accumulator_plaintext,
            ciphertext_modulus,
        );

        let mut buffers = ComputationBuffers::new();
        let fft = Fft::new(polynomial_size);
        let fft_view = fft.as_view();

        buffers.resize(
            programmable_bootstrap_lwe_ciphertext_mem_optimized_requirement::<u64>(
                glwe_size,
                polynomial_size,
                fft_view,
            )
            .unwrap()
            .unaligned_bytes_required(),
        );
        let stack = buffers.stack();

        let (mut local_accumulator_data, stack) = stack.collect_aligned(
            aligned_vec::CACHELINE_ALIGN,
            accumulator.as_ref().iter().copied(),
        );
        let mut local_accumulator = GlweCiphertextMutView::from_container(
            &mut *local_accumulator_data,
            polynomial_size,
            ciphertext_modulus,
        );
        let mut improved_lwe = lwe.clone();
        self.improve_small_lwe_for_blind_rotate(&mut improved_lwe, LutCountLog(0));

        gen_blind_rotate_local_assign(
            self.fourier_bsk.as_view(),
            local_accumulator.as_mut_view(),
            ModulusSwitchOffset(0),
            LutCountLog(0),
            improved_lwe.as_ref(),
            fft_view,
            stack,
        );

        GlweCiphertext::from_container(
            local_accumulator.as_ref().to_vec(),
            polynomial_size,
            ciphertext_modulus,
        )
    }

    fn shared_standard_br_carrier(
        &self,
        lwe: &LweCiphertext<Vec<u64>>,
    ) -> GlweCiphertext<Vec<u64>> {
        let glwe_size = self.params.glwe_dimension().to_glwe_size();
        let polynomial_size = self.params.polynomial_size();
        let ciphertext_modulus = self.params.ciphertext_modulus();
        let padding = polynomial_size.0 / FUSED_SELECTOR_LUT_COUNT;
        let base_mu = lowest_level_amplitude(self.params.cbs_base_log(), self.params.cbs_level());

        let mut accumulator = vec![0u64; polynomial_size.0];
        for coeff in accumulator.iter_mut().take(padding) {
            *coeff = base_mu;
        }

        let accumulator_plaintext = PlaintextList::from_container(accumulator);
        let accumulator = allocate_and_trivially_encrypt_new_glwe_ciphertext(
            glwe_size,
            &accumulator_plaintext,
            ciphertext_modulus,
        );

        let mut buffers = ComputationBuffers::new();
        let fft = Fft::new(polynomial_size);
        let fft_view = fft.as_view();

        buffers.resize(
            programmable_bootstrap_lwe_ciphertext_mem_optimized_requirement::<u64>(
                glwe_size,
                polynomial_size,
                fft_view,
            )
            .unwrap()
            .unaligned_bytes_required(),
        );
        let stack = buffers.stack();

        let (mut local_accumulator_data, stack) = stack.collect_aligned(
            aligned_vec::CACHELINE_ALIGN,
            accumulator.as_ref().iter().copied(),
        );
        let mut local_accumulator = GlweCiphertextMutView::from_container(
            &mut *local_accumulator_data,
            polynomial_size,
            ciphertext_modulus,
        );
        let mut improved_lwe = lwe.clone();
        self.improve_small_lwe_for_blind_rotate(&mut improved_lwe, LutCountLog(0));

        gen_blind_rotate_local_assign(
            self.fourier_bsk.as_view(),
            local_accumulator.as_mut_view(),
            ModulusSwitchOffset(0),
            LutCountLog(0),
            improved_lwe.as_ref(),
            fft_view,
            stack,
        );

        GlweCiphertext::from_container(
            local_accumulator.as_ref().to_vec(),
            polynomial_size,
            ciphertext_modulus,
        )
    }

    pub fn shared_standard_br_carrier_for_debug(
        &self,
        lwe: &LweCiphertext<Vec<u64>>,
    ) -> GlweCiphertext<Vec<u64>> {
        self.shared_standard_br_carrier(lwe)
    }

    pub fn weighted_selector_truth_polynomial_with_layout_for_debug(
        polynomial_size: PolynomialSize,
        selector_idx: usize,
        scale: u64,
        slot_start: usize,
        slot_stride: usize,
    ) -> Polynomial<Vec<u64>> {
        weighted_selector_truth_polynomial_with_layout(
            polynomial_size,
            selector_idx,
            scale,
            slot_start,
            slot_stride,
        )
    }

    pub fn multiply_glwe_by_sparse_clear_polynomial_for_debug(
        glwe: &GlweCiphertext<Vec<u64>>,
        clear_poly: &Polynomial<Vec<u64>>,
    ) -> GlweCiphertext<Vec<u64>> {
        multiply_glwe_by_sparse_clear_polynomial(glwe, clear_poly)
    }
}

fn selector_truth_polynomial(
    polynomial_size: PolynomialSize,
    selector_idx: usize,
) -> Polynomial<Vec<u64>> {
    selector_truth_polynomial_with_layout(
        polynomial_size,
        selector_idx,
        DENSE_STANDARD_BR_SLOT_START,
        1,
    )
}

fn selector_truth_polynomial_manylut8(
    polynomial_size: PolynomialSize,
    selector_idx: usize,
) -> Polynomial<Vec<u64>> {
    let padding = polynomial_size.0 / 8;
    let mut coeffs = vec![0u64; polynomial_size.0];
    for slot in 0..8usize {
        let bit = ((slot >> selector_idx) & 1) == 1;
        coeffs[slot * padding] = if bit { 1 } else { 0u64.wrapping_sub(1) };
    }
    Polynomial::from_container(coeffs)
}

fn selector_truth_polynomial_with_layout(
    polynomial_size: PolynomialSize,
    selector_idx: usize,
    slot_start: usize,
    slot_stride: usize,
) -> Polynomial<Vec<u64>> {
    let padding = polynomial_size.0 / FUSED_SELECTOR_LUT_COUNT;
    let mut coeffs = vec![0u64; polynomial_size.0];
    let table_size = 1usize << FUSED_SELECTOR_GROUP_BITS;
    const LOGICAL_SLOT_SHIFT: usize = 7;

    for table_idx in 0..table_size {
        let slot = slot_start + table_idx * slot_stride;
        let logical = (table_idx + LOGICAL_SLOT_SHIFT) % table_size;
        let bit = ((logical >> selector_idx) & 1) == 1;
        coeffs[slot * padding] = if bit { 1 } else { 0u64.wrapping_sub(1) };
    }

    Polynomial::from_container(coeffs)
}

fn weighted_selector_truth_polynomial_with_layout(
    polynomial_size: PolynomialSize,
    selector_idx: usize,
    scale: u64,
    slot_start: usize,
    slot_stride: usize,
) -> Polynomial<Vec<u64>> {
    let padding = polynomial_size.0 / FUSED_SELECTOR_LUT_COUNT;
    let mut coeffs = vec![0u64; polynomial_size.0];
    let table_size = 1usize << FUSED_SELECTOR_GROUP_BITS;

    for table_idx in 0..table_size {
        let slot_idx = (table_idx + table_size - 1) % table_size;
        let slot = slot_start + slot_idx * slot_stride;
        let bit = ((table_idx >> selector_idx) & 1) == 1;
        coeffs[slot * padding] = if bit { scale } else { 0u64.wrapping_sub(scale) };
    }

    Polynomial::from_container(coeffs)
}

fn multiply_glwe_by_clear_polynomial<Cont>(
    glwe: &GlweCiphertext<Cont>,
    clear_poly: &Polynomial<Vec<u64>>,
) -> GlweCiphertext<Vec<u64>>
where
    Cont: Container<Element = u64>,
{
    let mut out = GlweCiphertext::new(
        0u64,
        glwe.glwe_size(),
        glwe.polynomial_size(),
        glwe.ciphertext_modulus(),
    );

    for (mut dst_poly, src_poly) in out
        .as_mut_polynomial_list()
        .iter_mut()
        .zip(glwe.as_polynomial_list().iter())
    {
        polynomial_mul_by_fft::<u64, _, _, _>(&mut dst_poly, &src_poly, clear_poly);
    }

    out
}

fn multiply_glwe_by_sparse_clear_polynomial(
    glwe: &GlweCiphertext<Vec<u64>>,
    clear_poly: &Polynomial<Vec<u64>>,
) -> GlweCiphertext<Vec<u64>> {
    let mut out = GlweCiphertext::new(
        0u64,
        glwe.glwe_size(),
        glwe.polynomial_size(),
        glwe.ciphertext_modulus(),
    );

    let nonzero_terms = clear_poly
        .as_ref()
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, coeff)| *coeff != 0)
        .collect::<Vec<_>>();

    for (mut dst_poly, src_poly) in out
        .as_mut_polynomial_list()
        .iter_mut()
        .zip(glwe.as_polynomial_list().iter())
    {
        let mut rotated = Polynomial::from_container(vec![0u64; glwe.polynomial_size().0]);
        for (degree, coeff) in &nonzero_terms {
            polynomial_wrapping_monic_monomial_mul(
                &mut rotated,
                &src_poly,
                MonomialDegree(*degree),
            );
            for (dst, src) in dst_poly.as_mut().iter_mut().zip(rotated.as_ref().iter()) {
                *dst = dst.wrapping_add(src.wrapping_mul(*coeff));
            }
        }
    }

    out
}

pub fn packed_group_base_delta(num_bits: usize) -> u64 {
    assert!((1..64).contains(&num_bits), "num_bits must be in 1..64");
    1u64 << (64 - (num_bits + 1) - 1)
}

pub fn packed_group_bit_delta(num_bits: usize, bit_index: usize) -> u64 {
    assert!(
        bit_index < num_bits,
        "bit_index must be smaller than num_bits"
    );
    packed_group_base_delta(num_bits) << bit_index
}

pub fn fused_selector_group_centering_offset() -> u64 {
    let t_size = 1usize << FUSED_SELECTOR_GROUP_BITS;
    let slot_delta = packed_group_base_delta(FUSED_SELECTOR_GROUP_BITS);
    ((FUSED_SELECTOR_LUT_COUNT - t_size) as u64)
        .wrapping_mul(slot_delta)
        .wrapping_add(slot_delta >> 1)
        .wrapping_add(slot_delta >> 3)
}

pub fn encode_packed_boolean_group(bits: [bool; FUSED_SELECTOR_GROUP_BITS]) -> u64 {
    fused_selector_group_value(bits)
        .wrapping_mul(packed_group_base_delta(FUSED_SELECTOR_GROUP_BITS))
        .wrapping_add(fused_selector_group_centering_offset())
}

pub fn fused_selector_group_value(bits: [bool; FUSED_SELECTOR_GROUP_BITS]) -> u64 {
    bits.into_iter()
        .enumerate()
        .fold(0u64, |acc, (bit_index, bit)| {
            acc | ((bit as u64) << bit_index)
        })
}

pub fn encode_fused_selector_group(bits: [bool; FUSED_SELECTOR_GROUP_BITS]) -> u64 {
    encode_packed_boolean_group(bits)
}

pub fn encode_standard_br_boolean_group(bits: [bool; FUSED_SELECTOR_GROUP_BITS]) -> u64 {
    let weights = [1u64, 2, 4];
    let t_size = 1 << FUSED_SELECTOR_GROUP_BITS;
    let mut torus_message = 0u64;

    for (bit, &weight) in bits.into_iter().zip(weights.iter()) {
        let signed_mu = if bit {
            STANDARD_BR_LINEAR_BOOLEAN_MU * STANDARD_BR_LINEAR_GROUP_SCALE
        } else {
            0u64.wrapping_sub(STANDARD_BR_LINEAR_BOOLEAN_MU * STANDARD_BR_LINEAR_GROUP_SCALE)
        };
        torus_message = torus_message.wrapping_add(signed_mu.wrapping_mul(weight));
    }

    let bias = (2 * (STANDARD_BR_LINEAR_SLOT_COUNT - t_size)
        + weights.iter().copied().sum::<u64>() as usize
        + 1) as u64;

    torus_message.wrapping_add(
        (bias.wrapping_add(STANDARD_BR_LINEAR_SLOT_OFFSET_UNITS))
            .wrapping_mul(STANDARD_BR_LINEAR_BOOLEAN_MU),
    )
}

pub fn inter_group_standard_br_bit_delta(bit_index: usize) -> u64 {
    assert!(
        bit_index < FUSED_SELECTOR_GROUP_BITS,
        "bit_index must be smaller than the grouped selector width"
    );
    (standard_br_slot_delta().wrapping_mul(INTER_GROUP_STANDARD_BR_SLOT_STRIDE as u64)) << bit_index
}

pub fn inter_group_standard_br_base_offset() -> u64 {
    standard_br_group_base_offset(INTER_GROUP_STANDARD_BR_SLOT_START)
}

pub fn encode_inter_group_standard_br_value(packed_bits: u64) -> u64 {
    let mut encoded = inter_group_standard_br_base_offset();
    for bit_index in 0..FUSED_SELECTOR_GROUP_BITS {
        if ((packed_bits >> bit_index) & 1) == 1 {
            encoded = encoded.wrapping_add(inter_group_standard_br_bit_delta(bit_index));
        }
    }
    encoded
}

fn standard_br_slot_delta() -> u64 {
    1u64 << (64 - FUSED_SELECTOR_LUT_COUNT_LOG - 1)
}

fn standard_br_group_base_offset(slot_start: usize) -> u64 {
    let slot_delta = standard_br_slot_delta();
    (slot_start as u64)
        .wrapping_mul(slot_delta)
        .wrapping_add(slot_delta >> 1)
}

fn apply_standard_br_input_bias(value: u64) -> u64 {
    if STANDARD_BR_INPUT_BIAS == 0 {
        value
    } else if STANDARD_BR_INPUT_BIAS > 0 {
        value.wrapping_add(STANDARD_BR_INPUT_BIAS as u64)
    } else {
        value.wrapping_sub((-STANDARD_BR_INPUT_BIAS) as u64)
    }
}

fn apply_standard_br_input_bias_to_lwe(lwe: &mut LweCiphertext<Vec<u64>>) {
    if STANDARD_BR_INPUT_BIAS == 0 {
        return;
    }
    let offset = if STANDARD_BR_INPUT_BIAS > 0 {
        STANDARD_BR_INPUT_BIAS as u64
    } else {
        0u64.wrapping_sub((-STANDARD_BR_INPUT_BIAS) as u64)
    };
    lwe_ciphertext_plaintext_add_assign(lwe, Plaintext(offset));
}

fn level_amplitude(cbs_base_log: DecompositionBaseLog, level_idx: usize) -> u64 {
    1u64 << (64 - (level_idx + 1) * cbs_base_log.0 - 1)
}

fn lowest_level_amplitude(
    cbs_base_log: DecompositionBaseLog,
    cbs_level: DecompositionLevelCount,
) -> u64 {
    level_amplitude(cbs_base_log, cbs_level.0 - 1)
}

fn clone_fourier_ggsw(view: FourierGgswCiphertext<&[c64]>) -> SelectorCiphertext {
    let mut owned = FourierGgswCiphertext::new(
        view.glwe_size(),
        view.polynomial_size(),
        view.decomposition_base_log(),
        view.decomposition_level_count(),
    );
    owned.as_mut_view().data().copy_from_slice(view.data());
    owned
}

fn lwe_phase_unsigned_local(lwe: &LweCiphertext<Vec<u64>>, sk: &LweSecretKey<Vec<u64>>) -> u64 {
    let lwe_dim = sk.lwe_dimension().0;
    let slice = lwe.as_ref();
    let (mask, body_slice) = slice.split_at(lwe_dim);
    let body = body_slice[0];
    let mut dot: u64 = 0;
    for (&a_i, &s_i) in mask.iter().zip(sk.as_ref().iter()) {
        dot = dot.wrapping_add(a_i.wrapping_mul((s_i & 1) as u64));
    }
    body.wrapping_sub(dot)
}

fn center_u64_local(value: u64) -> i128 {
    if value >= (1u64 << 63) {
        (value as i128) - (1i128 << 64)
    } else {
        value as i128
    }
}

pub fn decode_torus(plaintext: u64, delta: u64) -> u64 {
    assert!(delta != 0, "delta must be non-zero");
    assert!(delta.is_power_of_two(), "delta must be a power of two");

    let rounded = (plaintext as u128 + (delta as u128 / 2)) / delta as u128;
    let message_bits = 64u32 - delta.trailing_zeros();

    if message_bits == 64 {
        rounded as u64
    } else {
        let message_modulus = 1u128 << message_bits;
        (rounded % message_modulus) as u64
    }
}
