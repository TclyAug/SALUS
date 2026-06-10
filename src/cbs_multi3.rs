use aligned_vec::CACHELINE_ALIGN;
use pulp::c64;
use refined_tfhe_lhe::{
    convert_lwe_to_glwe_const, gen_blind_rotate_local_assign, glwe_ciphertext_clone_from,
    glwe_ciphertext_monic_monomial_div_assign, lwe_preprocessing_assign, polynomial_mul_by_fft,
    switch_scheme, trace_assign,
};
use tfhe::core_crypto::{fft_impl::fft64::math::fft::Fft, prelude::*};

use crate::cbs::{CircuitBootstrapKeys, SelectorCiphertext};

pub const MULTI3_GROUP_BITS: usize = 3;
const SLOT_COUNT: usize = 8;
const MANYLUT_LOG_LUT_COUNT: usize = 1;
const BOOLEAN_MU: u64 = 1u64 << 58;
const LINEAR_GROUP_SCALE: u64 = 2;
const SLOT_OFFSET_UNITS: u64 = 6;

pub fn build_linear_small_lwe_multi3(
    keys: &CircuitBootstrapKeys,
    bits: [bool; MULTI3_GROUP_BITS],
) -> LweCiphertext<Vec<u64>> {
    let lwe_dimension = keys.small_lwe_secret_key.lwe_dimension();
    let weights = [1u64, 2, 4];
    let t_size = 1 << MULTI3_GROUP_BITS;

    let mut acc = LweCiphertext::new(
        0u64,
        lwe_dimension.to_lwe_size(),
        CiphertextModulus::<u64>::new_native(),
    );
    for (bit, &weight) in bits.into_iter().zip(weights.iter()) {
        let mut ct = LweCiphertext::new(
            0u64,
            lwe_dimension.to_lwe_size(),
            CiphertextModulus::<u64>::new_native(),
        );
        *ct.get_mut_body().data = if bit {
            BOOLEAN_MU * LINEAR_GROUP_SCALE
        } else {
            0u64.wrapping_sub(BOOLEAN_MU * LINEAR_GROUP_SCALE)
        };
        lwe_ciphertext_cleartext_mul_assign(&mut ct, Cleartext(weight));
        lwe_ciphertext_add_assign(&mut acc, &ct);
    }
    let bias =
        (2 * (SLOT_COUNT - t_size) + weights.iter().copied().sum::<u64>() as usize + 1) as u64;
    lwe_ciphertext_plaintext_add_assign(
        &mut acc,
        Plaintext((bias.wrapping_add(SLOT_OFFSET_UNITS)).wrapping_mul(BOOLEAN_MU)),
    );
    acc
}

pub fn pbs_manylut_multi3_levels(
    keys: &CircuitBootstrapKeys,
    small_lwe: &LweCiphertext<Vec<u64>>,
) -> Vec<GlweCiphertext<Vec<u64>>> {
    let glwe_size = keys.params.glwe_dimension().to_glwe_size();
    let polynomial_size = keys.params.polynomial_size();
    let ciphertext_modulus = keys.params.ciphertext_modulus();
    let padding = polynomial_size.0 / SLOT_COUNT;
    let cbs_base_log = keys.params.cbs_base_log();
    let cbs_level = keys.params.cbs_level();
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

    let (mut local_accumulator_data, stack) =
        stack.collect_aligned(CACHELINE_ALIGN, accumulator.as_ref().iter().copied());
    let mut local_accumulator = GlweCiphertextMutView::from_container(
        &mut *local_accumulator_data,
        polynomial_size,
        ciphertext_modulus,
    );
    let mut improved_small_lwe = small_lwe.clone();
    keys.improve_small_lwe_for_blind_rotate(
        &mut improved_small_lwe,
        LutCountLog(MANYLUT_LOG_LUT_COUNT),
    );

    gen_blind_rotate_local_assign(
        keys.fourier_bsk.as_view(),
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

    let mut outputs = Vec::with_capacity(MULTI3_GROUP_BITS * cbs_level.0);
    for selector_idx in 0..MULTI3_GROUP_BITS {
        let poly = selector_truth_poly_multi3(selector_idx, polynomial_size);
        let masked0 = multiply_glwe_by_clear_polynomial_multi3(&extracted[0], &poly);
        let masked1 = multiply_glwe_by_clear_polynomial_multi3(&extracted[1], &poly);
        outputs.push(masked0);
        outputs.push(masked1);
    }
    outputs
}

pub fn bootstrap_boolean_group_multi3(
    keys: &CircuitBootstrapKeys,
    bits: [bool; MULTI3_GROUP_BITS],
) -> Vec<SelectorCiphertext> {
    let small_lwe = build_linear_small_lwe_multi3(keys, bits);
    bootstrap_small_lwe_group_multi3(keys, &small_lwe)
}

pub fn bootstrap_small_lwe_group_multi3(
    keys: &CircuitBootstrapKeys,
    small_lwe: &LweCiphertext<Vec<u64>>,
) -> Vec<SelectorCiphertext> {
    let glwe_size = keys.params.glwe_dimension().to_glwe_size();
    let polynomial_size = keys.params.polynomial_size();
    let ciphertext_modulus = keys.params.ciphertext_modulus();
    let cbs_base_log = keys.params.cbs_base_log();
    let cbs_level = keys.params.cbs_level();
    let bootstrapped_levels = pbs_manylut_multi3_levels(keys, small_lwe);

    let mut selectors = Vec::with_capacity(MULTI3_GROUP_BITS);
    for selector_idx in 0..MULTI3_GROUP_BITS {
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
                keys.large_lwe_secret_key.lwe_dimension().to_lwe_size(),
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
            trace_assign(&mut glwe, &keys.auto_keys);
        }

        let mut ggsw = GgswCiphertext::new(
            0u64,
            glwe_size,
            polynomial_size,
            cbs_base_log,
            cbs_level,
            ciphertext_modulus,
        );
        switch_scheme(&glev, &mut ggsw, keys.scheme_switch_key.as_view());

        let mut fourier =
            FourierGgswCiphertext::new(glwe_size, polynomial_size, cbs_base_log, cbs_level);
        convert_standard_ggsw_ciphertext_to_fourier(&ggsw, &mut fourier);
        selectors.push(clone_fourier_ggsw_local(fourier.as_view()));
    }

    selectors
}

fn clone_fourier_ggsw_local(view: FourierGgswCiphertext<&[c64]>) -> SelectorCiphertext {
    let mut owned = FourierGgswCiphertext::new(
        view.glwe_size(),
        view.polynomial_size(),
        view.decomposition_base_log(),
        view.decomposition_level_count(),
    );
    owned.as_mut_view().data().copy_from_slice(view.data());
    owned
}

fn selector_truth_poly_multi3(
    selector_idx: usize,
    polynomial_size: PolynomialSize,
) -> Polynomial<Vec<u64>> {
    let pad = polynomial_size.0 / SLOT_COUNT;
    let mut coeffs = vec![0u64; polynomial_size.0];
    for slot in 0..SLOT_COUNT {
        let logical = slot;
        let bit = ((logical >> selector_idx) & 1) == 1;
        let sign = if bit { 1 } else { -1 };
        coeffs[slot * pad] = if sign > 0 { 1 } else { 0u64.wrapping_sub(1) };
    }
    Polynomial::from_container(coeffs)
}

fn multiply_glwe_by_clear_polynomial_multi3<Cont>(
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
