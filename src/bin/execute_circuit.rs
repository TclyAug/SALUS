use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::path::Path;
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

use refined_tfhe_lhe::int_lhe_instance::{
    SALUS_CMUX0 as CMUX0_EVAL_PARAM, SALUS_CMUX1 as CMUX1_EVAL_PARAM,
};
use salus::{
    encode_inter_group_standard_br_value, evaluate_bdd_with_refs,
    inter_group_standard_br_bit_delta, Bdd, CircuitBootstrapKeys, ImportedDagData,
    PreprocessedCircuit, SelectorCiphertext, FUSED_SELECTOR_GROUP_BITS,
};
use tfhe::core_crypto::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EvalRoute {
    Single,
    Hybrid,
    Multi,
}

impl EvalRoute {
    fn uses_grouped_path(self) -> bool {
        matches!(self, EvalRoute::Hybrid | EvalRoute::Multi)
    }

    fn uses_weighted_grouping(self) -> bool {
        matches!(self, EvalRoute::Multi)
    }
}

struct RunReport {
    actual_outputs: Vec<u64>,
    input_selector_bootstrap_ms: f64,
    bdd_eval_ms: f64,
    inter_lut_cb_ms: f64,
}

impl RunReport {
    fn pure_server_compute_ms(&self) -> f64 {
        self.input_selector_bootstrap_ms + self.bdd_eval_ms + self.inter_lut_cb_ms
    }
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_bit(&mut self) -> u64 {
        self.next_u64() & 1
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let dir_path = args
        .next()
        .ok_or("usage: execute_circuit <preprocessed_dir> [input_bits] [--seed N]")?;
    let mut optional_args = args.collect::<Vec<_>>();
    let requested_single_lut_mode = take_flag(&mut optional_args, "--single-lut-mode")
        || take_flag(&mut optional_args, "--single");
    let requested_hybrid_lut_mode = take_flag(&mut optional_args, "--hybrid-lut-mode")
        || take_flag(&mut optional_args, "--hybrid");
    let requested_multi_lut_mode = take_flag(&mut optional_args, "--multi-lut-mode")
        || take_flag(&mut optional_args, "--multi");
    let requested_fused_input_cb = take_flag(&mut optional_args, "--fused-input-cb");
    let requested_legacy_hybrid_grouped_inputs =
        take_flag(&mut optional_args, "--hybrid-cmux1-inputs");
    let requested_grouped_inter_cb = take_flag(&mut optional_args, "--grouped-inter-cb");
    let requested_packed_merge_groups = take_flag(&mut optional_args, "--packed-merge-groups");
    let requested_legacy_debug_scalar_eval = take_flag(&mut optional_args, "--debug-eval-cmux0");
    let fixed_seed = parse_u64_option(&mut optional_args, "--seed")?;
    let grouped_inter_level_limit =
        parse_usize_option(&mut optional_args, "--grouped-inter-level-limit")?;
    let repeat_random_count = parse_usize_option(&mut optional_args, "--repeat-random")?;
    let requested_mode_count = requested_single_lut_mode as usize
        + requested_hybrid_lut_mode as usize
        + requested_multi_lut_mode as usize;
    if requested_mode_count > 1 {
        return Err(
            "choose exactly one of --single-lut-mode, --hybrid-lut-mode, or --multi-lut-mode"
                .into(),
        );
    }
    if requested_fused_input_cb
        || requested_legacy_hybrid_grouped_inputs
        || requested_grouped_inter_cb
        || requested_packed_merge_groups
        || requested_legacy_debug_scalar_eval
    {
        return Err(
            "legacy routing flags have been removed; use --single-lut-mode, --hybrid-lut-mode, or --multi-lut-mode"
                .into(),
        );
    }
    let route = if requested_single_lut_mode {
        EvalRoute::Single
    } else if requested_hybrid_lut_mode {
        EvalRoute::Hybrid
    } else if requested_multi_lut_mode {
        EvalRoute::Multi
    } else {
        EvalRoute::Single
    };
    let use_grouped_path = route.uses_grouped_path();
    let use_grouped_input_cb = use_grouped_path;
    let use_weighted_grouped_inter_cb = route.uses_weighted_grouping();
    let use_packed_merge_groups = use_grouped_path;
    let use_scalar_singletons = matches!(route, EvalRoute::Single | EvalRoute::Hybrid);
    let use_grouped_singletons = matches!(route, EvalRoute::Multi);
    let circuit = PreprocessedCircuit::read_from_dir(Path::new(&dir_path))?;
    let dag_map = circuit.load_dags()?;
    let packed_group_bdds = if use_packed_merge_groups {
        let packed_groups = circuit
            .packed_groups
            .as_ref()
            .ok_or("packed merge groups requested, but packed_groups.txt is missing")?;
        Some(
            packed_groups
                .groups
                .iter()
                .map(|group| {
                    let encoded_truth_table = group
                        .packed_truth_table
                        .iter()
                        .map(|value| encode_inter_group_standard_br_value(*value))
                        .collect::<Vec<_>>();
                    Bdd::from_truth_table(group.inputs.len(), &encoded_truth_table)
                })
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let provided_input_bits = optional_args
        .first()
        .filter(|raw| !raw.starts_with("--"))
        .map(|raw| parse_input_bits(raw, circuit.inputs.len()))
        .transpose()?;
    if let Some(repeat_count) = repeat_random_count {
        if repeat_count == 0 {
            return Err("--repeat-random must be at least 1".into());
        }
        if provided_input_bits.is_some() {
            return Err(
                "--repeat-random cannot be combined with an explicit input bitstring".into(),
            );
        }
    }

    let delta = 1u64 << 63;
    let cmux1_base_delta = inter_group_standard_br_bit_delta(0);
    let cmux0_param = *CMUX0_EVAL_PARAM;
    let cmux1_param = *CMUX1_EVAL_PARAM;

    let cmux0_keys = if use_scalar_singletons {
        Some(if let Some(seed) = fixed_seed {
            CircuitBootstrapKeys::new_with_seed(cmux0_param, seed as u128)
        } else {
            CircuitBootstrapKeys::new(cmux0_param)
        })
    } else {
        None
    };
    let cmux1_keys = if use_grouped_path {
        Some(match route {
            EvalRoute::Multi => {
                if let Some(seed) = fixed_seed {
                    CircuitBootstrapKeys::new_with_seed(cmux1_param, seed as u128)
                } else {
                    CircuitBootstrapKeys::new(cmux1_param)
                }
            }
            EvalRoute::Hybrid => {
                let cmux0_keys = cmux0_keys
                    .as_ref()
                    .expect("hybrid route requires cmux0 keys");
                if let Some(seed) = fixed_seed {
                    CircuitBootstrapKeys::new_with_shared_glwe_secret_and_seed(
                        cmux1_param,
                        &cmux0_keys.glwe_secret_key,
                        seed as u128 ^ 0x4752_4f55_5045_445f_434d_5558_325fu128,
                    )
                } else {
                    CircuitBootstrapKeys::new_with_shared_glwe_secret(
                        cmux1_param,
                        &cmux0_keys.glwe_secret_key,
                    )
                }
            }
            EvalRoute::Single => unreachable!("single route does not build grouped keys"),
        })
    } else {
        None
    };
    let verify_keys = cmux0_keys
        .as_ref()
        .or(cmux1_keys.as_ref())
        .expect("at least one key set must be available");
    let run_once = |primary_input_bits: &[u64]| -> Result<RunReport, Box<dyn Error>> {
        let plain_signal_values =
            evaluate_preprocessed_plain(&circuit, &dag_map, primary_input_bits)?;
        let expected_outputs = circuit
            .outputs
            .iter()
            .map(|name| {
                plain_signal_values
                    .get(name)
                    .copied()
                    .ok_or_else(|| format!("missing plain output {name}"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let input_bootstrap_start = Instant::now();
        let mut live_signals = bootstrap_primary_inputs(
            cmux0_keys.as_ref(),
            cmux1_keys.as_ref(),
            &circuit,
            primary_input_bits,
            use_grouped_input_cb,
        )?;
        let input_selector_bootstrap_ms = input_bootstrap_start.elapsed().as_secs_f64() * 1000.0;

        let mut remaining_refs = circuit.ref_counts.clone();
        let output_names = circuit
            .outputs
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        let mut total_bdd_ms = 0.0;
        let mut total_cb_ms = 0.0;
        let mut final_output_glwes: HashMap<String, (GlweCiphertext<Vec<u64>>, u64)> =
            HashMap::with_capacity(circuit.outputs.len());

        for (level_number, level) in circuit.levels.iter().enumerate() {
            let grouping_enabled_this_level = grouped_inter_level_limit
                .map(|limit| level_number < limit)
                .unwrap_or(true);
            let packed_group_this_level = use_packed_merge_groups && grouping_enabled_this_level;
            let weighted_group_this_level =
                use_weighted_grouped_inter_cb && grouping_enabled_this_level;
            let mut packed_merged_lut_indices = HashSet::new();
            if packed_group_this_level {
                let packed_groups = circuit.packed_groups.as_ref().ok_or(
                    "packed merge groups requested, but packed group metadata is unavailable",
                )?;
                let packed_group_bdds = packed_group_bdds.as_ref().ok_or(
                    "packed merge groups requested, but packed group BDDs are unavailable",
                )?;

                for &group_idx in &packed_groups.levels[level_number] {
                    let group = &packed_groups.groups[group_idx];
                    let packed_bdd = &packed_group_bdds[group_idx];
                    let selector_inputs = group
                        .inputs
                        .iter()
                        .rev()
                        .map(|signal_name| {
                            live_signals
                                .get(signal_name)
                                .ok_or_else(|| format!("missing selector for {signal_name}"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    let bdd_start = Instant::now();
                    let packed_bdd_keys = if matches!(route, EvalRoute::Hybrid) {
                        cmux0_keys
                            .as_ref()
                            .expect("hybrid route requires cmux0 keys for packed BDD evaluation")
                    } else {
                        cmux1_keys
                            .as_ref()
                            .expect("grouped route requires cmux1 keys")
                    };
                    let packed_glwe =
                        evaluate_bdd_with_refs(packed_bdd, &selector_inputs, packed_bdd_keys, 1)?;
                    total_bdd_ms += bdd_start.elapsed().as_secs_f64() * 1000.0;

                    let cb_start = Instant::now();
                    let cmux1_keys = cmux1_keys
                        .as_ref()
                        .expect("grouped route requires cmux1 keys for packed multi-output CB");
                    let selectors = cmux1_keys.packed_glwe_to_selectors_standard_br(
                        &packed_glwe,
                        group.lut_indices.len(),
                    );
                    total_cb_ms += cb_start.elapsed().as_secs_f64() * 1000.0;

                    for (selector_idx, (&lut_index, selector)) in
                        group.lut_indices.iter().zip(selectors.iter()).enumerate()
                    {
                        let lut = &circuit.luts[lut_index];
                        let expected_value = plain_signal_values
                            .get(&lut.name)
                            .copied()
                            .ok_or_else(|| format!("missing plaintext value for {}", lut.name))?;
                        let actual_value = materialize_selector_bit(verify_keys, selector, delta);
                        if actual_value != expected_value {
                            let mut input_debug = Vec::with_capacity(group.inputs.len());
                            for input_name in &group.inputs {
                                let plain = plain_signal_values
                                    .get(input_name)
                                    .copied()
                                    .unwrap_or(u64::MAX);
                                let hom = live_signals.get(input_name).map(|input_selector| {
                                    materialize_selector_bit(verify_keys, input_selector, delta)
                                });
                                input_debug.push(format!(
                                    "{input_name}: plain={plain}, hom={}",
                                    hom.map(|value| value.to_string())
                                        .unwrap_or_else(|| "missing".to_string())
                                ));
                            }
                            return Err(format!(
                                "packed grouped selector mismatch for {} in group {} output {}: expected {}, got {}; inputs [{}]",
                                lut.name,
                                group_idx,
                                selector_idx,
                                expected_value,
                                actual_value,
                                input_debug.join(", ")
                            )
                            .into());
                        }

                        for input_name in &lut.inputs {
                            if let Some(remaining) = remaining_refs.get_mut(input_name) {
                                *remaining = remaining.saturating_sub(1);
                                if *remaining == 0 {
                                    live_signals.remove(input_name);
                                }
                            }
                        }

                        packed_merged_lut_indices.insert(lut_index);
                        live_signals.insert(lut.name.clone(), selector.clone());
                    }
                }
            }
            let remaining_level_luts = if packed_merged_lut_indices.is_empty() {
                level.clone()
            } else {
                level
                    .iter()
                    .copied()
                    .filter(|lut_index| !packed_merged_lut_indices.contains(lut_index))
                    .collect::<Vec<_>>()
            };
            let mut level_index = 0usize;
            while level_index < remaining_level_luts.len() {
                let remaining_level = remaining_level_luts.len() - level_index;
                let group_width = if weighted_group_this_level {
                    if remaining_level >= 2 {
                        remaining_level.min(FUSED_SELECTOR_GROUP_BITS)
                    } else {
                        0
                    }
                } else {
                    0
                };

                if group_width > 0 {
                    let mut grouped_names = Vec::with_capacity(group_width);
                    let mut grouped_glwes = Vec::with_capacity(group_width);
                    let mut grouped_expected_values = Vec::with_capacity(group_width);

                    for lut_index in remaining_level_luts[level_index..level_index + group_width]
                        .iter()
                        .copied()
                    {
                        let lut = &circuit.luts[lut_index];
                        let imported = dag_map
                            .get(&lut.name)
                            .ok_or_else(|| format!("missing DAG for {}", lut.name))?;
                        let bdd = Bdd::from_imported_boolean_dag(&imported.imported_bdd);
                        let expected_value = plain_signal_values
                            .get(&lut.name)
                            .copied()
                            .ok_or_else(|| format!("missing plaintext value for {}", lut.name))?;
                        let selector_inputs = lut
                            .inputs
                            .iter()
                            .rev()
                            .map(|signal_name| {
                                live_signals
                                    .get(signal_name)
                                    .ok_or_else(|| format!("missing selector for {signal_name}"))
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let bdd_start = Instant::now();
                        let cmux1_keys = cmux1_keys
                            .as_ref()
                            .expect("grouped route requires cmux1 keys");
                        let lut_glwe = evaluate_bdd_with_refs(
                            &bdd,
                            &selector_inputs,
                            cmux1_keys,
                            cmux1_base_delta,
                        )?;
                        total_bdd_ms += bdd_start.elapsed().as_secs_f64() * 1000.0;

                        let actual_value =
                            verify_keys.decrypt_glwe_coefficient0(&lut_glwe, cmux1_base_delta);
                        if actual_value != expected_value {
                            let mut input_debug = Vec::with_capacity(lut.inputs.len());
                            for input_name in &lut.inputs {
                                let plain = plain_signal_values
                                    .get(input_name)
                                    .copied()
                                    .unwrap_or(u64::MAX);
                                let hom = live_signals.get(input_name).map(|selector| {
                                    materialize_selector_bit(verify_keys, selector, delta)
                                });
                                input_debug.push(format!(
                                    "{input_name}: plain={plain}, hom={}",
                                    hom.map(|value| value.to_string())
                                        .unwrap_or_else(|| "missing".to_string())
                                ));
                            }
                            return Err(format!(
                                "LUT {} mismatch: expected {}, got {}; inputs [{}]",
                                lut.name,
                                expected_value,
                                actual_value,
                                input_debug.join(", ")
                            )
                            .into());
                        }

                        if output_names.contains(&lut.name) {
                            final_output_glwes
                                .insert(lut.name.clone(), (lut_glwe.clone(), cmux1_base_delta));
                        }

                        for input_name in &lut.inputs {
                            if let Some(remaining) = remaining_refs.get_mut(input_name) {
                                *remaining = remaining.saturating_sub(1);
                                if *remaining == 0 {
                                    live_signals.remove(input_name);
                                }
                            }
                        }

                        grouped_names.push(lut.name.clone());
                        grouped_glwes.push(lut_glwe);
                        grouped_expected_values.push(expected_value);
                    }

                    let cb_start = Instant::now();
                    let cmux1_keys = cmux1_keys
                        .as_ref()
                        .expect("grouped route requires cmux1 keys");
                    let selectors =
                        cmux1_keys.weighted_glwe_outputs_to_selectors_standard_br(&grouped_glwes);
                    total_cb_ms += cb_start.elapsed().as_secs_f64() * 1000.0;

                    for ((name, expected_value), selector) in grouped_names
                        .into_iter()
                        .zip(grouped_expected_values.into_iter())
                        .zip(selectors)
                    {
                        let actual_value = materialize_selector_bit(verify_keys, &selector, delta);
                        if actual_value != expected_value {
                            return Err(format!(
                                "grouped CB selector mismatch for {}: expected {}, got {}",
                                name, expected_value, actual_value
                            )
                            .into());
                        }
                        live_signals.insert(name, selector);
                    }

                    level_index += group_width;
                } else {
                    let lut_index = remaining_level_luts[level_index];
                    let lut = &circuit.luts[lut_index];
                    let imported = dag_map
                        .get(&lut.name)
                        .ok_or_else(|| format!("missing DAG for {}", lut.name))?;
                    let bdd = Bdd::from_imported_boolean_dag(&imported.imported_bdd);
                    let expected_value = plain_signal_values
                        .get(&lut.name)
                        .copied()
                        .ok_or_else(|| format!("missing plaintext value for {}", lut.name))?;
                    let selector_inputs = lut
                        .inputs
                        .iter()
                        .rev()
                        .map(|signal_name| {
                            live_signals
                                .get(signal_name)
                                .ok_or_else(|| format!("missing selector for {signal_name}"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    let bdd_start = Instant::now();
                    let singleton_bdd_keys = if use_grouped_singletons {
                        cmux1_keys
                            .as_ref()
                            .expect("grouped route requires cmux1 keys for singleton LUTs")
                    } else {
                        cmux0_keys
                            .as_ref()
                            .expect("single/hybrid route requires cmux0 keys for singleton LUTs")
                    };
                    let lut_glwe =
                        evaluate_bdd_with_refs(&bdd, &selector_inputs, singleton_bdd_keys, delta)?;
                    total_bdd_ms += bdd_start.elapsed().as_secs_f64() * 1000.0;

                    let actual_value = verify_keys.decrypt_glwe_coefficient0(&lut_glwe, delta);
                    if actual_value != expected_value {
                        let mut input_debug = Vec::with_capacity(lut.inputs.len());
                        for input_name in &lut.inputs {
                            let plain = plain_signal_values
                                .get(input_name)
                                .copied()
                                .unwrap_or(u64::MAX);
                            let hom = live_signals.get(input_name).map(|selector| {
                                materialize_selector_bit(verify_keys, selector, delta)
                            });
                            input_debug.push(format!(
                                "{input_name}: plain={plain}, hom={}",
                                hom.map(|value| value.to_string())
                                    .unwrap_or_else(|| "missing".to_string())
                            ));
                        }
                        return Err(format!(
                            "LUT {} mismatch: expected {}, got {}; inputs [{}]",
                            lut.name,
                            expected_value,
                            actual_value,
                            input_debug.join(", ")
                        )
                        .into());
                    }

                    if output_names.contains(&lut.name) {
                        final_output_glwes.insert(lut.name.clone(), (lut_glwe.clone(), delta));
                    }

                    for input_name in &lut.inputs {
                        if let Some(remaining) = remaining_refs.get_mut(input_name) {
                            *remaining = remaining.saturating_sub(1);
                            if *remaining == 0 {
                                live_signals.remove(input_name);
                            }
                        }
                    }

                    let cb_start = Instant::now();
                    let singleton_cb_keys = if use_grouped_singletons {
                        cmux1_keys
                            .as_ref()
                            .expect("multi route requires cmux1 keys for singleton CB")
                    } else {
                        cmux0_keys
                            .as_ref()
                            .expect("single/hybrid route requires cmux0 keys for singleton CB")
                    };
                    let selector = singleton_cb_keys.glwe_boolean_to_selector(&lut_glwe);
                    total_cb_ms += cb_start.elapsed().as_secs_f64() * 1000.0;
                    live_signals.insert(lut.name.clone(), selector);
                    level_index += 1;
                }
            }
        }

        let actual_outputs = circuit
            .outputs
            .iter()
            .map(|output_name| {
                if let Some(selector) = live_signals.get(output_name) {
                    Ok(materialize_selector_bit(verify_keys, selector, delta))
                } else if let Some((glwe, output_delta)) = final_output_glwes.get(output_name) {
                    Ok(verify_keys.decrypt_glwe_coefficient0(glwe, *output_delta))
                } else {
                    let selector = live_signals.get(output_name).ok_or_else(|| {
                        format!("missing final selector or GLWE for {output_name}")
                    })?;
                    Ok(materialize_selector_bit(verify_keys, selector, delta))
                }
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

        if actual_outputs != expected_outputs {
            return Err(format!(
                "circuit output mismatch: expected {}, got {}",
                format_bits(&expected_outputs),
                format_bits(&actual_outputs)
            )
            .into());
        }

        Ok(RunReport {
            actual_outputs,
            input_selector_bootstrap_ms,
            bdd_eval_ms: total_bdd_ms,
            inter_lut_cb_ms: total_cb_ms,
        })
    };

    if let Some(repeat_count) = repeat_random_count {
        let seed = fixed_seed.unwrap_or_else(now_seed);
        let mut rng = XorShift64::new(seed ^ 0x7265_7065_6174_7261_u64);
        let mut total_ms = 0.0;

        for run_idx in 0..repeat_count {
            let primary_input_bits = random_input_bits(circuit.inputs.len(), &mut rng);
            let report = run_once(&primary_input_bits)?;
            total_ms += report.pure_server_compute_ms();
            println!("run_{}:", run_idx + 1);
            println!("input_bits: {}", format_bits(&primary_input_bits));
            println!("output_bits: {}", format_bits(&report.actual_outputs));
            println!("pure_server_compute_ms: {:.2}", report.pure_server_compute_ms());
            println!("correct: true");
            if run_idx + 1 < repeat_count {
                println!();
            }
        }

        println!();
        let average_ms = total_ms / repeat_count as f64;
        println!("average_pure_server_compute_ms: {:.2}", average_ms);
        println!("correct: true");
    } else {
        let primary_input_bits =
            provided_input_bits.unwrap_or_else(|| vec![0; circuit.inputs.len()]);
        let report = run_once(&primary_input_bits)?;

        println!("input_bits: {}", format_bits(&primary_input_bits));
        println!("output_bits: {}", format_bits(&report.actual_outputs));
        println!("pure_server_compute_ms: {:.2}", report.pure_server_compute_ms());
        println!("correct: true");
    }
    Ok(())
}

fn evaluate_preprocessed_plain(
    circuit: &PreprocessedCircuit,
    dag_map: &HashMap<String, ImportedDagData>,
    primary_input_bits: &[u64],
) -> Result<HashMap<String, u64>, Box<dyn Error>> {
    if primary_input_bits.len() != circuit.inputs.len() {
        return Err(format!(
            "primary input width mismatch: expected {}, got {}",
            circuit.inputs.len(),
            primary_input_bits.len()
        )
        .into());
    }
    let mut values = HashMap::with_capacity(circuit.inputs.len() + circuit.luts.len());
    for (name, bit) in circuit
        .inputs
        .iter()
        .zip(primary_input_bits.iter().copied())
    {
        values.insert(name.clone(), bit);
    }
    for level in &circuit.levels {
        for &lut_index in level {
            let lut = &circuit.luts[lut_index];
            let assignment = lut.inputs.iter().enumerate().try_fold(
                0usize,
                |acc, (input_position, input_name)| {
                    let bit = values
                        .get(input_name)
                        .copied()
                        .ok_or_else(|| format!("missing signal {input_name}"))?;
                    Ok::<usize, Box<dyn Error>>(
                        acc | ((bit as usize)
                            << (lut.inputs.len().saturating_sub(1) - input_position)),
                    )
                },
            )?;
            let truth_table = &dag_map
                .get(&lut.name)
                .ok_or_else(|| format!("missing DAG for {}", lut.name))?
                .truth_table;
            values.insert(lut.name.clone(), truth_table[assignment]);
        }
    }
    Ok(values)
}

fn bootstrap_primary_inputs(
    cmux0_keys: Option<&CircuitBootstrapKeys>,
    cmux1_keys: Option<&CircuitBootstrapKeys>,
    circuit: &PreprocessedCircuit,
    primary_input_bits: &[u64],
    use_grouped_input_cb: bool,
) -> Result<HashMap<String, SelectorCiphertext>, Box<dyn Error>> {
    let mut live_signals = HashMap::with_capacity(circuit.inputs.len() + circuit.luts.len());
    let mut input_index = 0usize;
    while input_index < circuit.inputs.len() {
        let remaining = circuit.inputs.len() - input_index;
        if use_grouped_input_cb && remaining >= FUSED_SELECTOR_GROUP_BITS {
            let cmux1_keys = cmux1_keys.ok_or("grouped input CB requested without cmux1 keys")?;
            let bits = [
                primary_input_bits[input_index] == 1,
                primary_input_bits[input_index + 1] == 1,
                primary_input_bits[input_index + 2] == 1,
            ];
            let selectors = cmux1_keys.bootstrap_boolean_group_standard_br(bits);
            for (offset, selector) in selectors.into_iter().enumerate() {
                live_signals.insert(circuit.inputs[input_index + offset].clone(), selector);
            }
            input_index += FUSED_SELECTOR_GROUP_BITS;
        } else {
            let fallback_keys = cmux0_keys
                .or(cmux1_keys)
                .ok_or("scalar input CB requested without any available key set")?;
            let selector =
                fallback_keys.bootstrap_boolean_input(primary_input_bits[input_index] == 1);
            live_signals.insert(circuit.inputs[input_index].clone(), selector);
            input_index += 1;
        }
    }
    Ok(live_signals)
}

fn materialize_selector_bit(
    keys: &CircuitBootstrapKeys,
    selector: &SelectorCiphertext,
    delta: u64,
) -> u64 {
    let mut zero = keys.encrypt_glwe_constant(0, delta);
    let mut one = keys.encrypt_glwe_constant(1, delta);
    cmux_assign(&mut zero, &mut one, selector);
    keys.decrypt_glwe_coefficient0(&zero, delta)
}

fn parse_input_bits(raw: &str, expected_width: usize) -> Result<Vec<u64>, Box<dyn Error>> {
    let trimmed = raw.strip_prefix("0b").unwrap_or(raw);
    if trimmed.len() > expected_width {
        return Err(format!(
            "input bitstring too wide: expected at most {expected_width}, got {}",
            trimmed.len()
        )
        .into());
    }
    if !trimmed.bytes().all(|byte| byte == b'0' || byte == b'1') {
        return Err("input bitstring must contain only 0 or 1".into());
    }
    let mut padded = String::with_capacity(expected_width);
    for _ in 0..expected_width.saturating_sub(trimmed.len()) {
        padded.push('0');
    }
    padded.push_str(trimmed);
    Ok(padded
        .bytes()
        .map(|byte| if byte == b'1' { 1 } else { 0 })
        .collect())
}

fn parse_u64_option(args: &mut Vec<String>, flag: &str) -> Result<Option<u64>, Box<dyn Error>> {
    take_option(args, flag)
        .map(|value| value.parse::<u64>().map_err(|err| err.into()))
        .transpose()
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

fn format_bits(bits: &[u64]) -> String {
    bits.iter()
        .map(|bit| if *bit == 0 { '0' } else { '1' })
        .collect()
}

fn now_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0x1234_5678_9abc_def0)
}

fn random_input_bits(width: usize, rng: &mut XorShift64) -> Vec<u64> {
    (0..width).map(|_| rng.next_bit()).collect()
}
