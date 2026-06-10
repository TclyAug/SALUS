use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

use tfhe::core_crypto::prelude::*;

use crate::bdd::{Bdd, BddNode, NodeId};
use crate::cbs::{CircuitBootstrapKeys, SelectorCiphertext};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MuxTreeError {
    MissingSelector { variable_index: usize },
}

impl Display for MuxTreeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MuxTreeError::MissingSelector { variable_index } => {
                write!(f, "missing selector for BDD variable {variable_index}")
            }
        }
    }
}

impl Error for MuxTreeError {}

pub fn evaluate_bdd(
    bdd: &Bdd,
    selectors: &[SelectorCiphertext],
    keys: &CircuitBootstrapKeys,
    delta: u64,
) -> Result<GlweCiphertext<Vec<u64>>, MuxTreeError> {
    let selector_refs = selectors.iter().collect::<Vec<_>>();
    evaluate_bdd_with_refs(bdd, &selector_refs, keys, delta)
}

pub fn evaluate_bdd_with_refs(
    bdd: &Bdd,
    selectors: &[&SelectorCiphertext],
    keys: &CircuitBootstrapKeys,
    delta: u64,
) -> Result<GlweCiphertext<Vec<u64>>, MuxTreeError> {
    let mut terminal_cache = HashMap::new();
    let mut node_cache = HashMap::new();

    evaluate_node(
        bdd,
        bdd.root(),
        selectors,
        keys,
        delta,
        &mut terminal_cache,
        &mut node_cache,
    )
}

pub fn evaluate_truth_table(
    num_vars: usize,
    truth_table: &[u64],
    selectors: &[SelectorCiphertext],
    keys: &CircuitBootstrapKeys,
    delta: u64,
) -> Result<GlweCiphertext<Vec<u64>>, MuxTreeError> {
    let bdd = Bdd::from_truth_table(num_vars, truth_table);
    evaluate_bdd(&bdd, selectors, keys, delta)
}

pub fn evaluate_logic_fn<F>(
    num_vars: usize,
    logic_fn: F,
    selectors: &[SelectorCiphertext],
    keys: &CircuitBootstrapKeys,
    delta: u64,
) -> Result<GlweCiphertext<Vec<u64>>, MuxTreeError>
where
    F: FnMut(usize) -> u64,
{
    let bdd = Bdd::from_logic_fn(num_vars, logic_fn);
    evaluate_bdd(&bdd, selectors, keys, delta)
}

fn evaluate_node(
    bdd: &Bdd,
    node_id: NodeId,
    selectors: &[&SelectorCiphertext],
    keys: &CircuitBootstrapKeys,
    delta: u64,
    terminal_cache: &mut HashMap<u64, GlweCiphertext<Vec<u64>>>,
    node_cache: &mut HashMap<NodeId, GlweCiphertext<Vec<u64>>>,
) -> Result<GlweCiphertext<Vec<u64>>, MuxTreeError> {
    if let Some(ciphertext) = node_cache.get(&node_id) {
        return Ok(ciphertext.clone());
    }

    let ciphertext = match bdd.node(node_id) {
        BddNode::Terminal(value) => {
            if let Some(ciphertext) = terminal_cache.get(value) {
                ciphertext.clone()
            } else {
                // Terminal values are public constants, so a trivial GLWE avoids
                // injecting fresh noise before the CMUX tree starts.
                let ciphertext = keys.trivially_encrypt_glwe_constant(*value, delta);
                terminal_cache.insert(*value, ciphertext.clone());
                ciphertext
            }
        }
        BddNode::Branch {
            variable_index,
            low,
            high,
        } => {
            let selector = selectors
                .get(*variable_index)
                .ok_or(MuxTreeError::MissingSelector {
                    variable_index: *variable_index,
                })?;
            let mut low_ciphertext = evaluate_node(
                bdd,
                *low,
                selectors,
                keys,
                delta,
                terminal_cache,
                node_cache,
            )?;
            let mut high_ciphertext = evaluate_node(
                bdd,
                *high,
                selectors,
                keys,
                delta,
                terminal_cache,
                node_cache,
            )?;
            cmux_assign(&mut low_ciphertext, &mut high_ciphertext, selector);
            low_ciphertext
        }
    };

    node_cache.insert(node_id, ciphertext.clone());
    Ok(ciphertext)
}
