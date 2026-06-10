use tfhe::core_crypto::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct NoiseEstimationMeasureBound(pub f64);

#[derive(Clone, Copy, Debug)]
pub struct RSigmaFactor(pub f64);

pub fn improve_lwe_ciphertext_modulus_switch_noise_for_binary_key(
    lwe: &mut LweCiphertext<Vec<u64>>,
    zero_encryptions: &[LweCiphertext<Vec<u64>>],
    input_variance_modular: f64,
    log_modulus: CiphertextModulusLog,
    acceptable_noise_estimation: NoiseEstimationMeasureBound,
    r_sigma_factor: RSigmaFactor,
) {
    assert!(
        lwe.ciphertext_modulus().is_native_modulus(),
        "modulus-switch noise reduction expects native-modulus LWE ciphertexts",
    );

    let mut best_index = None;
    let mut best_measure = f64::INFINITY;

    for (index, enc0) in zero_encryptions.iter().enumerate() {
        assert_eq!(
            lwe.lwe_size(),
            enc0.lwe_size(),
            "zero-encryption LWE size must match the input LWE size",
        );
        assert!(
            enc0.ciphertext_modulus().is_native_modulus(),
            "zero-encryption candidates must use the native modulus",
        );

        let current_measure = measure_modulus_switch_noise_estimation_for_binary_key(
            lwe,
            enc0,
            input_variance_modular,
            log_modulus,
            r_sigma_factor,
        );

        if current_measure < best_measure {
            best_measure = current_measure;
            best_index = Some(index);
            if current_measure <= acceptable_noise_estimation.0 {
                break;
            }
        }
    }

    if let Some(best_index) = best_index {
        lwe_ciphertext_add_assign(lwe, &zero_encryptions[best_index]);
    }
}

fn measure_modulus_switch_noise_estimation_for_binary_key(
    lwe: &LweCiphertext<Vec<u64>>,
    zero_encryption: &LweCiphertext<Vec<u64>>,
    input_variance_modular: f64,
    log_modulus: CiphertextModulusLog,
    r_sigma_factor: RSigmaFactor,
) -> f64 {
    let lwe_size = lwe.lwe_size().0;
    let mask_size = lwe_size - 1;

    let input = lwe.as_ref();
    let enc0 = zero_encryption.as_ref();

    let body_rounding_error =
        round_error_modular(input[mask_size].wrapping_add(enc0[mask_size]), log_modulus);

    let mut mask_expectancy = 0.0f64;
    let mut mask_variance = 0.0f64;
    for (input_coeff, enc0_coeff) in input[..mask_size].iter().zip(enc0[..mask_size].iter()) {
        let rounding_error =
            round_error_modular(input_coeff.wrapping_add(*enc0_coeff), log_modulus);
        mask_expectancy += 0.5 * rounding_error;
        mask_variance += 0.25 * rounding_error * rounding_error;
    }

    let noise_expectancy = body_rounding_error + mask_expectancy;
    let noise_std_dev = (mask_variance + input_variance_modular).sqrt();

    noise_expectancy.abs() + r_sigma_factor.0 * noise_std_dev
}

fn round_error_modular(input: u64, log_modulus: CiphertextModulusLog) -> f64 {
    centered_torus_difference(round_to_native_torus_grid(input, log_modulus), input) as f64
}

fn round_to_native_torus_grid(input: u64, log_modulus: CiphertextModulusLog) -> u64 {
    if log_modulus.0 >= u64::BITS as usize {
        return input;
    }

    let shift = (u64::BITS as usize) - log_modulus.0;
    if shift == 0 {
        input
    } else {
        (((input >> (shift - 1)).wrapping_add(1)) >> 1) << shift
    }
}

fn centered_torus_difference(lhs: u64, rhs: u64) -> i128 {
    center_u64(lhs.wrapping_sub(rhs))
}

fn center_u64(value: u64) -> i128 {
    let half_modulus = 1u64 << 63;
    if value < half_modulus {
        value as i128
    } else {
        (value as i128) - (1i128 << 64)
    }
}
