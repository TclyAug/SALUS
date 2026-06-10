use std::error::Error;
use std::fs;
use std::path::Path;

use crate::bdd::{ImportedBdd, ImportedBranchNode, ImportedEdge};

#[derive(Clone, Debug)]
pub struct ImportedDagData {
    pub imported_bdd: ImportedBdd,
    pub truth_table: Vec<u64>,
}

pub fn read_cudd_dag_file(path: &Path) -> Result<ImportedDagData, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    parse_cudd_dag_str(&content)
}

pub fn parse_cudd_dag_str(content: &str) -> Result<ImportedDagData, Box<dyn Error>> {
    let mut num_vars = None;
    let mut root = None;
    let mut nodes = Vec::new();
    let mut truth_table = None;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["num_vars", n] => {
                num_vars = Some(n.parse::<usize>()?);
            }
            ["truth_table", bits] => {
                truth_table = Some(parse_truth_table(bits)?);
            }
            ["root", node_id, complemented] => {
                root = Some(parse_edge(node_id, complemented)?);
            }
            ["node", id, variable_index, low_id, low_complemented, high_id, high_complemented] => {
                nodes.push(ImportedBranchNode {
                    id: id.parse::<usize>()?,
                    variable_index: variable_index.parse::<usize>()?,
                    low: parse_edge(low_id, low_complemented)?,
                    high: parse_edge(high_id, high_complemented)?,
                });
            }
            _ => {
                return Err(format!("unrecognized DAG line: {line}").into());
            }
        }
    }

    Ok(ImportedDagData {
        imported_bdd: ImportedBdd {
            num_vars: num_vars.ok_or("missing num_vars line")?,
            root: root.ok_or("missing root line")?,
            nodes,
        },
        truth_table: truth_table.ok_or("missing truth_table line")?,
    })
}

fn parse_truth_table(bits: &str) -> Result<Vec<u64>, Box<dyn Error>> {
    bits.chars()
        .map(|bit| match bit {
            '0' => Ok(0),
            '1' => Ok(1),
            _ => Err(format!("invalid truth table bit: {bit}").into()),
        })
        .collect()
}

fn parse_edge(node_id: &str, complemented: &str) -> Result<ImportedEdge, Box<dyn Error>> {
    let complemented = match complemented {
        "0" => false,
        "1" => true,
        _ => return Err(format!("invalid complemented flag: {complemented}").into()),
    };

    Ok(ImportedEdge {
        node_id: if node_id == "-1" {
            None
        } else {
            Some(node_id.parse::<usize>()?)
        },
        complemented,
    })
}
