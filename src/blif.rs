use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::imported_dag::{parse_cudd_dag_str, ImportedDagData};

#[derive(Debug)]
pub enum BlifError {
    Io(std::io::Error),
    Parse(String),
    MissingSignal { signal: String },
    InvalidPrimaryInputWidth { expected: usize, actual: usize },
    MixedOutputPolarity { lut_name: String },
    CuddToolFailed { message: String },
}

impl Display for BlifError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Parse(message) => write!(f, "{message}"),
            Self::MissingSignal { signal } => write!(f, "missing signal {signal}"),
            Self::InvalidPrimaryInputWidth { expected, actual } => {
                write!(
                    f,
                    "primary input width mismatch: expected {expected} bits, got {actual}"
                )
            }
            Self::MixedOutputPolarity { lut_name } => {
                write!(
                    f,
                    "LUT {lut_name} uses mixed 0/1 cover rows, which is not supported yet"
                )
            }
            Self::CuddToolFailed { message } => write!(f, "{message}"),
        }
    }
}

impl Error for BlifError {}

impl From<std::io::Error> for BlifError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlifLut {
    pub name: String,
    pub inputs: Vec<String>,
    pub truth_table: Vec<u64>,
}

impl BlifLut {
    pub fn assignment_from_signal_values(
        &self,
        signal_values: &HashMap<String, u64>,
    ) -> Result<usize, BlifError> {
        let mut assignment = 0usize;
        let width = self.inputs.len();

        for (input_position, input_name) in self.inputs.iter().enumerate() {
            let bit =
                signal_values
                    .get(input_name)
                    .copied()
                    .ok_or_else(|| BlifError::MissingSignal {
                        signal: input_name.clone(),
                    })?;
            assignment |= (bit as usize) << (width.saturating_sub(1) - input_position);
        }

        Ok(assignment)
    }

    pub fn selector_signal_order(&self) -> impl Iterator<Item = &str> {
        self.inputs.iter().rev().map(|name| name.as_str())
    }

    pub fn truth_table_bits(&self) -> String {
        self.truth_table
            .iter()
            .map(|bit| if *bit == 0 { '0' } else { '1' })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlifCircuit {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub luts: Vec<BlifLut>,
    pub levels: Vec<Vec<usize>>,
    pub ref_counts: HashMap<String, usize>,
}

impl BlifCircuit {
    pub fn parse_file(path: &Path) -> Result<Self, BlifError> {
        let content = fs::read_to_string(path)?;
        Self::parse_str(&content)
    }

    pub fn parse_str(content: &str) -> Result<Self, BlifError> {
        let logical_lines = join_blif_lines(content);
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut luts = Vec::new();
        let mut line_index = 0usize;

        while line_index < logical_lines.len() {
            let line = &logical_lines[line_index];
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                line_index += 1;
                continue;
            }

            match parts[0] {
                ".model" => {
                    line_index += 1;
                }
                ".inputs" => {
                    inputs.extend(parts[1..].iter().map(|name| (*name).to_string()));
                    line_index += 1;
                }
                ".outputs" => {
                    outputs.extend(parts[1..].iter().map(|name| (*name).to_string()));
                    line_index += 1;
                }
                ".names" => {
                    if parts.len() < 2 {
                        return Err(BlifError::Parse(format!("invalid .names line: {line}")));
                    }

                    let node_inputs = parts[1..parts.len() - 1]
                        .iter()
                        .map(|name| (*name).to_string())
                        .collect::<Vec<_>>();
                    let node_output = parts[parts.len() - 1].to_string();
                    let mut entries = Vec::new();
                    line_index += 1;

                    while line_index < logical_lines.len()
                        && !logical_lines[line_index].starts_with('.')
                    {
                        let row = logical_lines[line_index].trim();
                        if row.is_empty() {
                            line_index += 1;
                            continue;
                        }

                        let row_parts: Vec<&str> = row.split_whitespace().collect();
                        if row_parts.is_empty() {
                            line_index += 1;
                            continue;
                        }

                        let pattern = row_parts[0].to_string();
                        let output = row_parts
                            .get(1)
                            .and_then(|raw| raw.chars().next())
                            .unwrap_or('1');
                        entries.push((pattern, output));
                        line_index += 1;
                    }

                    let truth_table = build_truth_table(&node_output, node_inputs.len(), &entries)?;
                    luts.push(BlifLut {
                        name: node_output,
                        inputs: node_inputs,
                        truth_table,
                    });
                }
                ".end" => break,
                _ => {
                    line_index += 1;
                }
            }
        }

        let ref_counts = compute_ref_counts(&inputs, &outputs, &luts);
        let levels = topological_levels(&inputs, &luts)?;

        Ok(Self {
            inputs,
            outputs,
            luts,
            levels,
            ref_counts,
        })
    }

    pub fn evaluate_primary_input_bits(
        &self,
        primary_input_bits: &[u64],
    ) -> Result<HashMap<String, u64>, BlifError> {
        if primary_input_bits.len() != self.inputs.len() {
            return Err(BlifError::InvalidPrimaryInputWidth {
                expected: self.inputs.len(),
                actual: primary_input_bits.len(),
            });
        }

        let mut signal_values = HashMap::with_capacity(self.inputs.len() + self.luts.len());
        for (input_name, bit) in self.inputs.iter().zip(primary_input_bits.iter().copied()) {
            signal_values.insert(input_name.clone(), bit);
        }

        for level in &self.levels {
            for &lut_index in level {
                let lut = &self.luts[lut_index];
                let assignment = lut.assignment_from_signal_values(&signal_values)?;
                let value = lut.truth_table[assignment];
                signal_values.insert(lut.name.clone(), value);
            }
        }

        Ok(signal_values)
    }

    pub fn output_values(
        &self,
        signal_values: &HashMap<String, u64>,
    ) -> Result<Vec<u64>, BlifError> {
        self.outputs
            .iter()
            .map(|output_name| {
                signal_values
                    .get(output_name)
                    .copied()
                    .ok_or_else(|| BlifError::MissingSignal {
                        signal: output_name.clone(),
                    })
            })
            .collect()
    }
}

pub fn reduce_lut_with_cudd(tool_path: &Path, lut: &BlifLut) -> Result<ImportedDagData, BlifError> {
    let truth_table_bits = lut.truth_table_bits();
    let output = if truth_table_bits.len() > 131_072 {
        let temp_path = std::env::temp_dir().join(format!(
            "salus_tt_{}_{}_{}.txt",
            lut.name,
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|err| BlifError::CuddToolFailed {
                    message: format!("failed to create temp filename for LUT {}: {err}", lut.name),
                })?
                .as_nanos()
        ));
        fs::write(&temp_path, &truth_table_bits)?;
        let output = Command::new(tool_path)
            .arg(lut.inputs.len().to_string())
            .arg(format!("@{}", temp_path.display()))
            .output()?;
        let _ = fs::remove_file(&temp_path);
        output
    } else {
        Command::new(tool_path)
            .arg(lut.inputs.len().to_string())
            .arg(&truth_table_bits)
            .output()?
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(BlifError::CuddToolFailed {
            message: format!(
                "cudd_tt_to_dag failed for LUT {}: {}",
                lut.name,
                stderr.trim()
            ),
        });
    }

    let stdout = String::from_utf8(output.stdout).map_err(|err| BlifError::CuddToolFailed {
        message: format!(
            "invalid UTF-8 from cudd_tt_to_dag for LUT {}: {err}",
            lut.name
        ),
    })?;

    parse_cudd_dag_str(&stdout).map_err(|err| BlifError::CuddToolFailed {
        message: format!("failed to parse CUDD DAG for LUT {}: {err}", lut.name),
    })
}

fn join_blif_lines(content: &str) -> Vec<String> {
    let mut logical_lines = Vec::new();
    let mut current = String::new();

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if current.is_empty() {
            current.push_str(line);
        } else {
            current.push(' ');
            current.push_str(line);
        }

        if current.ends_with('\\') {
            current.pop();
            while current.ends_with(' ') {
                current.pop();
            }
            continue;
        }

        logical_lines.push(std::mem::take(&mut current));
    }

    if !current.is_empty() {
        logical_lines.push(current);
    }

    logical_lines
}

fn build_truth_table(
    lut_name: &str,
    num_inputs: usize,
    entries: &[(String, char)],
) -> Result<Vec<u64>, BlifError> {
    if num_inputs == 0 {
        return Ok(match entries.first().map(|(_, output)| *output) {
            Some('1') => vec![1],
            _ => vec![0],
        });
    }

    let table_size = 1usize << num_inputs;
    let has_one_rows = entries.iter().any(|(_, output)| *output == '1');
    let has_zero_rows = entries.iter().any(|(_, output)| *output == '0');

    if has_one_rows && has_zero_rows {
        return Err(BlifError::MixedOutputPolarity {
            lut_name: lut_name.to_string(),
        });
    }

    let default_output = if has_zero_rows { 1u64 } else { 0u64 };
    let written_output = if has_zero_rows { 0u64 } else { 1u64 };
    let mut truth_table = vec![default_output; table_size];

    for (pattern, _) in entries {
        if pattern.len() != num_inputs {
            return Err(BlifError::Parse(format!(
                "truth table row width mismatch in LUT {lut_name}: expected {num_inputs}, got {}",
                pattern.len()
            )));
        }
        expand_pattern(pattern, 0, 0, num_inputs, written_output, &mut truth_table)?;
    }

    Ok(truth_table)
}

fn expand_pattern(
    pattern: &str,
    position: usize,
    assignment: usize,
    num_inputs: usize,
    output: u64,
    truth_table: &mut [u64],
) -> Result<(), BlifError> {
    if position == pattern.len() {
        truth_table[assignment] = output;
        return Ok(());
    }

    match pattern.as_bytes()[position] {
        b'0' => expand_pattern(
            pattern,
            position + 1,
            assignment,
            num_inputs,
            output,
            truth_table,
        )?,
        b'1' => expand_pattern(
            pattern,
            position + 1,
            assignment | (1usize << (num_inputs - 1 - position)),
            num_inputs,
            output,
            truth_table,
        )?,
        b'-' => {
            expand_pattern(
                pattern,
                position + 1,
                assignment,
                num_inputs,
                output,
                truth_table,
            )?;
            expand_pattern(
                pattern,
                position + 1,
                assignment | (1usize << (num_inputs - 1 - position)),
                num_inputs,
                output,
                truth_table,
            )?;
        }
        other => {
            return Err(BlifError::Parse(format!(
                "invalid BLIF pattern byte {}",
                other as char
            )));
        }
    }

    Ok(())
}

fn compute_ref_counts(
    inputs: &[String],
    outputs: &[String],
    luts: &[BlifLut],
) -> HashMap<String, usize> {
    let mut ref_counts = HashMap::with_capacity(inputs.len() + luts.len());

    for input_name in inputs {
        ref_counts.insert(input_name.clone(), 0);
    }
    for lut in luts {
        ref_counts.insert(lut.name.clone(), 0);
    }

    for lut in luts {
        for input_name in &lut.inputs {
            *ref_counts.entry(input_name.clone()).or_insert(0) += 1;
        }
    }

    for output_name in outputs {
        *ref_counts.entry(output_name.clone()).or_insert(0) += 1;
    }

    ref_counts
}

fn topological_levels(
    primary_inputs: &[String],
    luts: &[BlifLut],
) -> Result<Vec<Vec<usize>>, BlifError> {
    let primary_input_set = primary_inputs.iter().cloned().collect::<HashSet<_>>();
    let mut available_signals = primary_input_set.clone();
    let mut scheduled = HashSet::new();
    let mut levels = Vec::new();

    loop {
        let current_level = luts
            .iter()
            .enumerate()
            .filter(|(lut_index, lut)| {
                !scheduled.contains(lut_index)
                    && lut
                        .inputs
                        .iter()
                        .all(|input_name| available_signals.contains(input_name))
            })
            .map(|(lut_index, _)| lut_index)
            .collect::<Vec<_>>();

        if current_level.is_empty() {
            break;
        }

        for &lut_index in &current_level {
            scheduled.insert(lut_index);
            available_signals.insert(luts[lut_index].name.clone());
        }

        levels.push(current_level);
    }

    if scheduled.len() != luts.len() {
        return Err(BlifError::Parse(
            "BLIF network is not acyclic or references undefined signals".to_string(),
        ));
    }

    Ok(levels)
}

#[cfg(test)]
mod tests {
    use super::{build_truth_table, BlifCircuit};

    #[test]
    fn parses_off_set_cover_used_by_abc() {
        let truth_table = build_truth_table("out", 2, &[("00".to_string(), '0')]).unwrap();

        assert_eq!(truth_table, vec![0, 1, 1, 1]);
    }

    #[test]
    fn parses_small_two_lut_circuit() {
        let circuit = BlifCircuit::parse_str(
            "
            .model toy
            .inputs a b c
            .outputs y
            .names a b n1
            01 1
            10 1
            .names n1 c y
            11 1
            .end
            ",
        )
        .unwrap();

        assert_eq!(circuit.luts.len(), 2);
        assert_eq!(circuit.levels.len(), 2);
    }

    #[test]
    fn evaluates_small_two_lut_circuit() {
        let circuit = BlifCircuit::parse_str(
            "
            .model toy
            .inputs a b c
            .outputs y
            .names a b n1
            01 1
            10 1
            .names n1 c y
            11 1
            .end
            ",
        )
        .unwrap();

        let signal_values = circuit.evaluate_primary_input_bits(&[1, 0, 1]).unwrap();
        let outputs = circuit.output_values(&signal_values).unwrap();
        assert_eq!(outputs, vec![1]);
    }
}
