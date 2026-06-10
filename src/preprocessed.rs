use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use crate::blif::BlifCircuit;
use crate::imported_dag::{read_cudd_dag_file, ImportedDagData};
use crate::packed_groups::PackedLutGroup;

#[derive(Clone, Debug)]
pub struct PreprocessedLut {
    pub name: String,
    pub inputs: Vec<String>,
    pub truth_table_bits: String,
    pub dag_path: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreprocessedPackedGroup {
    pub lut_indices: Vec<usize>,
    pub inputs: Vec<String>,
    pub packed_truth_table: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreprocessedPackedGroups {
    pub levels: Vec<Vec<usize>>,
    pub groups: Vec<PreprocessedPackedGroup>,
}

#[derive(Clone, Debug)]
pub struct PreprocessedCircuit {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub levels: Vec<Vec<usize>>,
    pub ref_counts: HashMap<String, usize>,
    pub luts: Vec<PreprocessedLut>,
    pub packed_groups: Option<PreprocessedPackedGroups>,
}

impl PreprocessedCircuit {
    pub fn write_to_dir(&self, dir: &Path) -> Result<(), Box<dyn Error>> {
        fs::create_dir_all(dir)?;
        let mut text = String::new();
        text.push_str("version 1\n");
        text.push_str(&format!("inputs {}\n", self.inputs.len()));
        for name in &self.inputs {
            text.push_str(&format!("input {name}\n"));
        }
        text.push_str(&format!("outputs {}\n", self.outputs.len()));
        for name in &self.outputs {
            text.push_str(&format!("output {name}\n"));
        }
        text.push_str(&format!("levels {}\n", self.levels.len()));
        for level in &self.levels {
            let body = level
                .iter()
                .map(|idx| idx.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            text.push_str(&format!("level {body}\n"));
        }
        text.push_str(&format!("ref_counts {}\n", self.ref_counts.len()));
        for (name, count) in &self.ref_counts {
            text.push_str(&format!("ref {name} {count}\n"));
        }
        text.push_str(&format!("luts {}\n", self.luts.len()));
        for lut in &self.luts {
            text.push_str(&format!("lut {}\n", lut.name));
            text.push_str(&format!("lut_inputs {}\n", lut.inputs.len()));
            for input in &lut.inputs {
                text.push_str(&format!("lut_input {input}\n"));
            }
            text.push_str(&format!("truth_table {}\n", lut.truth_table_bits));
            text.push_str(&format!("dag_file {}\n", lut.dag_path.display()));
        }
        fs::write(dir.join("circuit.txt"), text)?;
        Ok(())
    }

    pub fn read_from_dir(dir: &Path) -> Result<Self, Box<dyn Error>> {
        let content = fs::read_to_string(dir.join("circuit.txt"))?;
        let mut lines = content.lines();

        let version = lines.next().ok_or("missing version line")?;
        if version != "version 1" {
            return Err(format!("unsupported format: {version}").into());
        }

        let inputs_count = parse_header_count(lines.next(), "inputs")?;
        let mut inputs = Vec::with_capacity(inputs_count);
        for _ in 0..inputs_count {
            inputs.push(parse_single_value(lines.next(), "input")?.to_string());
        }

        let outputs_count = parse_header_count(lines.next(), "outputs")?;
        let mut outputs = Vec::with_capacity(outputs_count);
        for _ in 0..outputs_count {
            outputs.push(parse_single_value(lines.next(), "output")?.to_string());
        }

        let levels_count = parse_header_count(lines.next(), "levels")?;
        let mut levels = Vec::with_capacity(levels_count);
        for _ in 0..levels_count {
            let line = lines.next().ok_or("missing level line")?;
            let rest = line
                .strip_prefix("level")
                .ok_or("invalid level prefix")?
                .trim();
            let level = if rest.is_empty() {
                Vec::new()
            } else {
                rest.split_whitespace()
                    .map(|s| s.parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()?
            };
            levels.push(level);
        }

        let refs_count = parse_header_count(lines.next(), "ref_counts")?;
        let mut ref_counts = HashMap::with_capacity(refs_count);
        for _ in 0..refs_count {
            let line = lines.next().ok_or("missing ref line")?;
            let mut parts = line.split_whitespace();
            if parts.next() != Some("ref") {
                return Err(format!("invalid ref line: {line}").into());
            }
            let name = parts.next().ok_or("missing ref name")?.to_string();
            let count = parts.next().ok_or("missing ref count")?.parse::<usize>()?;
            ref_counts.insert(name, count);
        }

        let lut_count = parse_header_count(lines.next(), "luts")?;
        let mut luts = Vec::with_capacity(lut_count);
        for _ in 0..lut_count {
            let name = parse_single_value(lines.next(), "lut")?.to_string();
            let input_count = parse_header_count(lines.next(), "lut_inputs")?;
            let mut lut_inputs = Vec::with_capacity(input_count);
            for _ in 0..input_count {
                lut_inputs.push(parse_single_value(lines.next(), "lut_input")?.to_string());
            }
            let truth_table_bits = parse_single_value(lines.next(), "truth_table")?.to_string();
            let dag_file = parse_single_value(lines.next(), "dag_file")?;
            luts.push(PreprocessedLut {
                name,
                inputs: lut_inputs,
                truth_table_bits,
                dag_path: dir.join(dag_file),
            });
        }

        Ok(Self {
            inputs,
            outputs,
            levels,
            ref_counts,
            luts,
            packed_groups: read_packed_groups_file(&dir.join("packed_groups.txt"))?,
        })
    }

    pub fn load_dags(&self) -> Result<HashMap<String, ImportedDagData>, Box<dyn Error>> {
        let mut dags = HashMap::with_capacity(self.luts.len());
        for lut in &self.luts {
            dags.insert(lut.name.clone(), read_cudd_dag_file(&lut.dag_path)?);
        }
        Ok(dags)
    }
}

pub fn from_blif_circuit_with_dags(
    circuit: &BlifCircuit,
    dag_paths: &[PathBuf],
) -> PreprocessedCircuit {
    let luts = circuit
        .luts
        .iter()
        .zip(dag_paths.iter())
        .map(|(lut, dag_path)| PreprocessedLut {
            name: lut.name.clone(),
            inputs: lut.inputs.clone(),
            truth_table_bits: lut.truth_table_bits(),
            dag_path: dag_path.clone(),
        })
        .collect();

    PreprocessedCircuit {
        inputs: circuit.inputs.clone(),
        outputs: circuit.outputs.clone(),
        levels: circuit.levels.clone(),
        ref_counts: circuit.ref_counts.clone(),
        luts,
        packed_groups: None,
    }
}

pub fn write_packed_groups_to_dir(
    dir: &Path,
    levels: &[Vec<PackedLutGroup>],
) -> Result<(), Box<dyn Error>> {
    let mut flat_groups = Vec::new();
    let mut level_group_indices = Vec::with_capacity(levels.len());

    for level in levels {
        let mut group_indices = Vec::new();
        for group in level {
            if group.lut_indices.len() <= 1 {
                continue;
            }
            group_indices.push(flat_groups.len());
            flat_groups.push(PreprocessedPackedGroup {
                lut_indices: group.lut_indices.clone(),
                inputs: group.inputs.clone(),
                packed_truth_table: group.packed_truth_table.clone(),
            });
        }
        level_group_indices.push(group_indices);
    }

    let mut text = String::new();
    text.push_str("version 1\n");
    text.push_str(&format!("levels {}\n", level_group_indices.len()));
    for level in &level_group_indices {
        let body = level
            .iter()
            .map(|idx| idx.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        text.push_str(&format!("level {body}\n"));
    }
    text.push_str(&format!("groups {}\n", flat_groups.len()));
    for group in &flat_groups {
        text.push_str(&format!("group_luts {}\n", group.lut_indices.len()));
        for lut_index in &group.lut_indices {
            text.push_str(&format!("lut_idx {lut_index}\n"));
        }
        text.push_str(&format!("group_inputs {}\n", group.inputs.len()));
        for input in &group.inputs {
            text.push_str(&format!("input {input}\n"));
        }
        let packed_truth_table = group
            .packed_truth_table
            .iter()
            .map(|value| {
                char::from_digit(*value as u32, 16)
                    .expect("packed truth table entry must fit into a single hex digit")
            })
            .collect::<String>();
        text.push_str(&format!("packed_truth_table {packed_truth_table}\n"));
    }
    fs::write(dir.join("packed_groups.txt"), text)?;
    Ok(())
}

fn parse_header_count(line: Option<&str>, prefix: &str) -> Result<usize, Box<dyn Error>> {
    let line = line.ok_or_else(|| format!("missing {prefix} line"))?;
    let mut parts = line.split_whitespace();
    if parts.next() != Some(prefix) {
        return Err(format!("invalid {prefix} line: {line}").into());
    }
    Ok(parts.next().ok_or("missing count")?.parse::<usize>()?)
}

fn parse_single_value<'a>(line: Option<&'a str>, prefix: &str) -> Result<&'a str, Box<dyn Error>> {
    let line = line.ok_or_else(|| format!("missing {prefix} line"))?;
    let mut parts = line.split_whitespace();
    if parts.next() != Some(prefix) {
        return Err(format!("invalid {prefix} line: {line}").into());
    }
    parts
        .next()
        .ok_or_else(|| format!("missing {prefix} value").into())
}

fn read_packed_groups_file(
    path: &Path,
) -> Result<Option<PreprocessedPackedGroups>, Box<dyn Error>> {
    if !path.exists() {
        return Ok(None);
    }

    let content = fs::read_to_string(path)?;
    let mut lines = content.lines();

    let version = lines.next().ok_or("missing packed_groups version line")?;
    if version != "version 1" {
        return Err(format!("unsupported packed_groups format: {version}").into());
    }

    let levels_count = parse_header_count(lines.next(), "levels")?;
    let mut levels = Vec::with_capacity(levels_count);
    for _ in 0..levels_count {
        let line = lines.next().ok_or("missing packed_groups level line")?;
        let rest = line
            .strip_prefix("level")
            .ok_or("invalid packed_groups level prefix")?
            .trim();
        let level = if rest.is_empty() {
            Vec::new()
        } else {
            rest.split_whitespace()
                .map(|s| s.parse::<usize>())
                .collect::<Result<Vec<_>, _>>()?
        };
        levels.push(level);
    }

    let groups_count = parse_header_count(lines.next(), "groups")?;
    let mut groups = Vec::with_capacity(groups_count);
    for _ in 0..groups_count {
        let lut_count = parse_header_count(lines.next(), "group_luts")?;
        let mut lut_indices = Vec::with_capacity(lut_count);
        for _ in 0..lut_count {
            lut_indices.push(parse_single_value(lines.next(), "lut_idx")?.parse::<usize>()?);
        }

        let input_count = parse_header_count(lines.next(), "group_inputs")?;
        let mut inputs = Vec::with_capacity(input_count);
        for _ in 0..input_count {
            inputs.push(parse_single_value(lines.next(), "input")?.to_string());
        }

        let packed_truth_table = parse_single_value(lines.next(), "packed_truth_table")?
            .chars()
            .map(|digit| {
                digit
                    .to_digit(16)
                    .map(|value| value as u64)
                    .ok_or_else(|| format!("invalid packed truth table digit: {digit}"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        groups.push(PreprocessedPackedGroup {
            lut_indices,
            inputs,
            packed_truth_table,
        });
    }

    Ok(Some(PreprocessedPackedGroups { levels, groups }))
}
