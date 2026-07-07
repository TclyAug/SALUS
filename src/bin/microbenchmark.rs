use std::error::Error;
use std::time::Instant;

use refined_tfhe_lhe::int_lhe_instance::{
    SALUS_CMUX0 as CMUX0_EVAL_PARAM, SALUS_CMUX1 as CMUX1_EVAL_PARAM,
};
use salus::{
    evaluate_bdd_with_refs_timed, Bdd, CircuitBootstrapKeys, ComponentTimingStats,
    SelectorCiphertext, TimedStat,
};
use tfhe::core_crypto::prelude::*;

const DEFAULT_REPEAT: usize = 1000;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let repeat = parse_usize_option(&mut args, "--repeat")?.unwrap_or(DEFAULT_REPEAT);
    if repeat == 0 {
        return Err("--repeat must be at least 1".into());
    }

    let use_cmux1 = take_flag(&mut args, "--cmux1");
    let param = if use_cmux1 {
        *CMUX1_EVAL_PARAM
    } else {
        *CMUX0_EVAL_PARAM
    };
    let param_name = if use_cmux1 {
        "SALUS_CMUX1"
    } else {
        "SALUS_CMUX0"
    };

    let keys = CircuitBootstrapKeys::new_with_seed(param, 0x5341_4c55_535f_4d49_4352_4f42);
    let delta = 1u64 << 63;
    let mut input_stats = ComponentTimingStats::default();
    let mut bdd_stats = ComponentTimingStats::default();
    let mut sample_extract_stats = ComponentTimingStats::default();
    let mut refresh_stats = ComponentTimingStats::default();

    let truth_table = xor3_truth_table();
    let bdd = Bdd::from_truth_table(3, &truth_table);
    let selectors = [
        keys.bootstrap_boolean_input(false),
        keys.bootstrap_boolean_input(true),
        keys.bootstrap_boolean_input(false),
    ];
    let selector_refs = selectors.iter().collect::<Vec<&SelectorCiphertext>>();

    let mut sample_glwe = keys.encrypt_glwe_constant(1, delta);
    let mut sample_lwe = keys.encrypt_boolean_input(true);

    for iter in 0..repeat {
        let bit = iter % 2 == 1;

        let input_start = Instant::now();
        let selector = keys.bootstrap_boolean_input_timed(bit, &mut input_stats);
        input_stats.record_input_scalar_bootstrap(input_start.elapsed());

        let bdd_start = Instant::now();
        sample_glwe =
            evaluate_bdd_with_refs_timed(&bdd, &selector_refs, &keys, delta, &mut bdd_stats)?;
        bdd_stats.record_bdd_tree(bdd_start.elapsed(), bdd.branch_count());

        let _ = keys.extract_coefficient0_as_lwe_timed(&sample_glwe, &mut sample_extract_stats);

        let refresh_start = Instant::now();
        let refreshed = keys.glwe_boolean_to_selector_timed(&sample_glwe, &mut refresh_stats);
        refresh_stats.record_singleton_refresh(refresh_start.elapsed());

        sample_lwe = keys.encrypt_boolean_input(!bit);
        sample_glwe = materialize_selector_as_glwe(&keys, &selector, delta);
        let _ = materialize_selector_as_glwe(&keys, &refreshed, delta);
    }

    // Keep the compiler from proving the setup values unused across optimized runs.
    let _ = keys.decrypt_large_lwe(&sample_lwe, delta);
    let _ = keys.decrypt_glwe_coefficient0(&sample_glwe, delta);

    println!("microbenchmark:");
    println!("param_set: {param_name}");
    println!("repeat: {repeat}");
    println!("bdd_inputs: {}", bdd.num_vars());
    println!("bdd_branch_nodes: {}", bdd.branch_count());
    println!();
    print_item(
        "input LWE->RLev",
        "input pre-conversion/bootstrap",
        &input_stats.input_scalar_bootstrap,
        Some(("ms_per_input", input_stats.input_scalar_bootstrap.mean_ms())),
    );
    print_item(
        "RLev->RGSW Conversion",
        "GLev/RLev to GGSW switch_scheme inside selector CBS",
        &input_stats.conversion,
        Some(("ms_per_input", input_stats.conversion.mean_ms())),
    );
    print_item(
        "single CMUX",
        "one cmux_assign operation",
        &bdd_stats.cmux,
        None,
    );
    print_item(
        "BDD eval",
        "one complete LUT CMUX tree",
        &bdd_stats.bdd_tree,
        Some((
            "ms_per_bdd_branch_node",
            bdd_stats.bdd_tree.total_ms() / bdd_stats.bdd_nodes.max(1) as f64,
        )),
    );
    print_item(
        "SampleExtract",
        "RLWE/GLWE to LWE coefficient-0 extraction",
        &sample_extract_stats.sample_extract,
        Some((
            "ms_per_extract",
            sample_extract_stats.sample_extract.mean_ms(),
        )),
    );
    print_item(
        "Refresh BR",
        "LWE to RLev selector refresh, includes KS/BR/conversion",
        &refresh_stats.singleton_refresh,
        Some(("ms_per_refresh", refresh_stats.singleton_refresh.mean_ms())),
    );

    println!();
    println!("refresh_breakdown:");
    print_raw_stat("sample_extract_rlwe_to_lwe", &refresh_stats.sample_extract);
    print_raw_stat("keyswitch_large_to_small", &refresh_stats.key_switch);
    print_raw_stat("blind_rotate_pbs_manylut", &refresh_stats.blind_rotate);
    print_raw_stat("conversion_glev_to_ggsw", &refresh_stats.conversion);
    print_raw_stat("fourier_ggsw_conversion", &refresh_stats.fourier_conversion);

    Ok(())
}

fn xor3_truth_table() -> Vec<u64> {
    (0usize..8)
        .map(|assignment| ((assignment.count_ones() & 1) != 0) as u64)
        .collect()
}

fn materialize_selector_as_glwe(
    keys: &CircuitBootstrapKeys,
    selector: &SelectorCiphertext,
    delta: u64,
) -> GlweCiphertext<Vec<u64>> {
    let mut zero = keys.encrypt_glwe_constant(0, delta);
    let mut one = keys.encrypt_glwe_constant(1, delta);
    cmux_assign(&mut zero, &mut one, selector);
    zero
}

fn print_item(name: &str, description: &str, stat: &TimedStat, extra: Option<(&str, f64)>) {
    print!(
        "{}: description=\"{}\" count={} total_ms={:.6} mean_ms={:.6} variance_ms2={:.6}",
        name,
        description,
        stat.count,
        stat.total_ms(),
        stat.mean_ms(),
        stat.variance_ms2()
    );
    if let Some((label, value)) = extra {
        print!(" {label}={value:.6}");
    }
    println!();
}

fn print_raw_stat(name: &str, stat: &TimedStat) {
    println!(
        "{name}: count={} total_ms={:.6} mean_ms={:.6} variance_ms2={:.6}",
        stat.count,
        stat.total_ms(),
        stat.mean_ms(),
        stat.variance_ms2()
    );
}

fn parse_usize_option(args: &mut Vec<String>, flag: &str) -> Result<Option<usize>, Box<dyn Error>> {
    take_option(args, flag)
        .map(|value| value.parse::<usize>().map_err(|err| err.into()))
        .transpose()
}

fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    if let Some(index) = args.iter().position(|arg| arg == flag) {
        args.remove(index);
        true
    } else {
        false
    }
}

fn take_option(args: &mut Vec<String>, flag: &str) -> Option<String> {
    args.iter().position(|arg| arg == flag).map(|index| {
        args.remove(index);
        args.remove(index)
    })
}
