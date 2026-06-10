use std::collections::{BTreeSet, HashMap, HashSet};

use crate::blif::BlifCircuit;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedLutGroup {
    pub lut_indices: Vec<usize>,
    pub output_names: Vec<String>,
    pub inputs: Vec<String>,
    pub packed_truth_table: Vec<u64>,
}

pub fn build_merge_lut2_groups(
    circuit: &BlifCircuit,
    max_inputs: usize,
    max_group_size: usize,
) -> Vec<Vec<PackedLutGroup>> {
    circuit
        .levels
        .iter()
        .map(|level| {
            let grouped_indices = group_level_indices(circuit, level, max_inputs, max_group_size);
            grouped_indices
                .into_iter()
                .map(|lut_indices| build_packed_group(circuit, &lut_indices))
                .collect()
        })
        .collect()
}

fn group_level_indices(
    circuit: &BlifCircuit,
    level: &[usize],
    max_inputs: usize,
    max_group_size: usize,
) -> Vec<Vec<usize>> {
    let input_sets = level
        .iter()
        .map(|&lut_index| {
            let inputs = circuit.luts[lut_index]
                .inputs
                .iter()
                .cloned()
                .collect::<HashSet<_>>();
            (lut_index, inputs)
        })
        .collect::<HashMap<_, _>>();

    let sorted_level = {
        let mut scored = level
            .iter()
            .map(|&lut_index| {
                let overlap_score = level
                    .iter()
                    .copied()
                    .filter(|other| *other != lut_index)
                    .map(|other| {
                        input_sets[&lut_index]
                            .intersection(&input_sets[&other])
                            .count()
                    })
                    .sum::<usize>();
                (lut_index, overlap_score)
            })
            .collect::<Vec<_>>();
        scored.sort_by(|(lhs_idx, lhs_score), (rhs_idx, rhs_score)| {
            rhs_score.cmp(lhs_score).then_with(|| lhs_idx.cmp(rhs_idx))
        });
        scored
            .into_iter()
            .map(|(lut_index, _)| lut_index)
            .collect::<Vec<_>>()
    };

    let mut groups: Vec<Vec<usize>> = Vec::new();
    for lut_index in sorted_level {
        let mut placed = false;
        for group in &mut groups {
            if group.len() >= max_group_size {
                continue;
            }

            let mut group_inputs = BTreeSet::new();
            for existing in group.iter().copied() {
                group_inputs.extend(circuit.luts[existing].inputs.iter().cloned());
            }
            group_inputs.extend(circuit.luts[lut_index].inputs.iter().cloned());

            if group_inputs.len() <= max_inputs {
                group.push(lut_index);
                placed = true;
                break;
            }
        }

        if !placed {
            groups.push(vec![lut_index]);
        }
    }

    groups
}

pub fn build_packed_group(circuit: &BlifCircuit, lut_indices: &[usize]) -> PackedLutGroup {
    assert!(
        !lut_indices.is_empty(),
        "packed group must contain at least one LUT"
    );
    assert!(
        lut_indices.len() <= u64::BITS as usize,
        "packed group width exceeds supported terminal packing width"
    );

    let inputs = lut_indices
        .iter()
        .flat_map(|lut_index| circuit.luts[*lut_index].inputs.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let table_size = 1usize << inputs.len();
    let input_positions = inputs
        .iter()
        .enumerate()
        .map(|(position, name)| (name.clone(), position))
        .collect::<HashMap<_, _>>();

    let packed_truth_table = (0..table_size)
        .map(|assignment| {
            lut_indices
                .iter()
                .enumerate()
                .fold(0u64, |packed, (bit_index, lut_index)| {
                    let lut = &circuit.luts[*lut_index];
                    let local_assignment = lut.inputs.iter().fold(0usize, |acc, input_name| {
                        let input_position = input_positions[input_name];
                        let bit = (assignment >> (inputs.len() - 1 - input_position)) & 1;
                        (acc << 1) | bit
                    });
                    packed | (lut.truth_table[local_assignment] << bit_index)
                })
        })
        .collect::<Vec<_>>();

    PackedLutGroup {
        lut_indices: lut_indices.to_vec(),
        output_names: lut_indices
            .iter()
            .map(|lut_index| circuit.luts[*lut_index].name.clone())
            .collect(),
        inputs,
        packed_truth_table,
    }
}

#[cfg(test)]
mod tests {
    use super::{build_merge_lut2_groups, build_packed_group};
    use crate::blif::BlifCircuit;

    #[test]
    fn packs_two_same_input_luts_into_shared_truth_table() {
        let circuit = BlifCircuit::parse_str(
            r#"
.model top
.inputs a b
.outputs x y
.names a b x
01 1
10 1
.names a b y
10 1
11 1
.end
"#,
        )
        .unwrap();

        let group = build_packed_group(&circuit, &[0, 1]);
        assert_eq!(group.output_names, vec!["x".to_string(), "y".to_string()]);
        assert_eq!(group.inputs, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(group.packed_truth_table, vec![0, 1, 3, 2]);
    }

    #[test]
    fn merge_lut2_grouping_respects_input_budget_and_group_width() {
        let circuit = BlifCircuit::parse_str(
            r#"
.model top
.inputs a b c d e
.outputs x y z
.names a b x
11 1
.names a c y
11 1
.names d e z
11 1
.end
"#,
        )
        .unwrap();

        let levels = build_merge_lut2_groups(&circuit, 3, 3);
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].len(), 2);
        assert_eq!(levels[0][0].lut_indices.len(), 2);
        assert_eq!(levels[0][1].lut_indices.len(), 1);
    }
}
