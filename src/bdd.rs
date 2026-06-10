use std::collections::HashMap;

pub type NodeId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImportedEdge {
    pub node_id: Option<NodeId>,
    pub complemented: bool,
}

impl ImportedEdge {
    pub fn node(node_id: NodeId) -> Self {
        Self {
            node_id: Some(node_id),
            complemented: false,
        }
    }

    pub fn complemented_node(node_id: NodeId) -> Self {
        Self {
            node_id: Some(node_id),
            complemented: true,
        }
    }

    pub fn terminal_one() -> Self {
        Self {
            node_id: None,
            complemented: false,
        }
    }

    pub fn terminal_zero() -> Self {
        Self {
            node_id: None,
            complemented: true,
        }
    }

    fn with_xor_complement(self, complemented: bool) -> Self {
        Self {
            node_id: self.node_id,
            complemented: self.complemented ^ complemented,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedBranchNode {
    pub id: NodeId,
    pub variable_index: usize,
    pub low: ImportedEdge,
    pub high: ImportedEdge,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedBdd {
    pub num_vars: usize,
    pub root: ImportedEdge,
    pub nodes: Vec<ImportedBranchNode>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BddNode {
    Terminal(u64),
    Branch {
        variable_index: usize,
        low: NodeId,
        high: NodeId,
    },
}

#[derive(Clone, Debug)]
pub struct Bdd {
    nodes: Vec<BddNode>,
    root: NodeId,
    num_vars: usize,
}

impl Bdd {
    pub fn from_truth_table(num_vars: usize, truth_table: &[u64]) -> Self {
        let variable_order: Vec<usize> = (0..num_vars).rev().collect();
        Self::from_truth_table_with_var_order(num_vars, truth_table, &variable_order)
    }

    pub fn from_logic_fn<F>(num_vars: usize, mut logic_fn: F) -> Self
    where
        F: FnMut(usize) -> u64,
    {
        let table_size = expected_table_len(num_vars);
        let truth_table: Vec<u64> = (0..table_size)
            .map(|assignment| logic_fn(assignment))
            .collect();
        Self::from_truth_table(num_vars, &truth_table)
    }

    pub fn from_imported_boolean_dag(imported: &ImportedBdd) -> Self {
        let mut node_by_id = HashMap::with_capacity(imported.nodes.len());
        for node in &imported.nodes {
            assert!(
                node.variable_index < imported.num_vars,
                "imported node variable index out of range"
            );
            let previous = node_by_id.insert(node.id, node);
            assert!(previous.is_none(), "duplicate imported node id");
        }

        let mut builder = BddBuilder::default();
        let mut cache = HashMap::new();
        let root = import_boolean_edge(imported.root, &node_by_id, &mut builder, &mut cache);

        Self {
            nodes: builder.nodes,
            root,
            num_vars: imported.num_vars,
        }
    }

    pub fn from_truth_table_with_var_order(
        num_vars: usize,
        truth_table: &[u64],
        variable_order: &[usize],
    ) -> Self {
        let table_size = expected_table_len(num_vars);
        assert_eq!(
            truth_table.len(),
            table_size,
            "truth table length must be exactly 2^num_vars"
        );
        assert_eq!(
            variable_order.len(),
            num_vars,
            "variable order length must equal num_vars"
        );

        let mut seen = vec![false; num_vars];
        for &variable_index in variable_order {
            assert!(variable_index < num_vars, "variable index out of range");
            assert!(!seen[variable_index], "variable order contains duplicates");
            seen[variable_index] = true;
        }

        let mut builder = BddBuilder::default();
        let root = builder.build_subtree(truth_table, variable_order);

        Self {
            nodes: builder.nodes,
            root,
            num_vars,
        }
    }

    pub fn evaluate(&self, assignment: usize) -> u64 {
        let mut current = self.root;

        loop {
            match self.node(current) {
                BddNode::Terminal(value) => return *value,
                BddNode::Branch {
                    variable_index,
                    low,
                    high,
                } => {
                    let bit = (assignment >> variable_index) & 1;
                    current = if bit == 0 { *low } else { *high };
                }
            }
        }
    }

    pub fn root(&self) -> NodeId {
        self.root
    }

    pub fn node(&self, node_id: NodeId) -> &BddNode {
        &self.nodes[node_id]
    }

    pub fn nodes(&self) -> &[BddNode] {
        &self.nodes
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn branch_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| matches!(node, BddNode::Branch { .. }))
            .count()
    }

    pub fn terminal_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| matches!(node, BddNode::Terminal(_)))
            .count()
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn max_variable_index(&self) -> Option<usize> {
        self.nodes
            .iter()
            .filter_map(|node| match node {
                BddNode::Terminal(_) => None,
                BddNode::Branch { variable_index, .. } => Some(*variable_index),
            })
            .max()
    }
}

#[derive(Default)]
struct BddBuilder {
    nodes: Vec<BddNode>,
    terminal_cache: HashMap<u64, NodeId>,
    unique_table: HashMap<(usize, NodeId, NodeId), NodeId>,
}

impl BddBuilder {
    fn build_subtree(&mut self, truth_table: &[u64], variable_order: &[usize]) -> NodeId {
        if truth_table.iter().all(|value| *value == truth_table[0]) {
            return self.terminal(truth_table[0]);
        }

        if variable_order.is_empty() {
            return self.terminal(truth_table[0]);
        }

        let split = truth_table.len() / 2;
        let low = self.build_subtree(&truth_table[..split], &variable_order[1..]);
        let high = self.build_subtree(&truth_table[split..], &variable_order[1..]);
        self.branch(variable_order[0], low, high)
    }

    fn terminal(&mut self, value: u64) -> NodeId {
        if let Some(existing) = self.terminal_cache.get(&value) {
            return *existing;
        }

        let node_id = self.nodes.len();
        self.nodes.push(BddNode::Terminal(value));
        self.terminal_cache.insert(value, node_id);
        node_id
    }

    fn branch(&mut self, variable_index: usize, low: NodeId, high: NodeId) -> NodeId {
        if low == high {
            return low;
        }

        let key = (variable_index, low, high);
        if let Some(existing) = self.unique_table.get(&key) {
            return *existing;
        }

        let node_id = self.nodes.len();
        self.nodes.push(BddNode::Branch {
            variable_index,
            low,
            high,
        });
        self.unique_table.insert(key, node_id);
        node_id
    }
}

fn expected_table_len(num_vars: usize) -> usize {
    1usize
        .checked_shl(num_vars as u32)
        .expect("num_vars is too large for usize indexing")
}

fn import_boolean_edge(
    edge: ImportedEdge,
    node_by_id: &HashMap<NodeId, &ImportedBranchNode>,
    builder: &mut BddBuilder,
    cache: &mut HashMap<ImportedEdge, NodeId>,
) -> NodeId {
    if let Some(existing) = cache.get(&edge) {
        return *existing;
    }

    let node_id = match edge.node_id {
        None => builder.terminal(if edge.complemented { 0 } else { 1 }),
        Some(imported_node_id) => {
            let imported = node_by_id
                .get(&imported_node_id)
                .unwrap_or_else(|| panic!("missing imported node id {imported_node_id}"));

            let low = import_boolean_edge(
                imported.low.with_xor_complement(edge.complemented),
                node_by_id,
                builder,
                cache,
            );
            let high = import_boolean_edge(
                imported.high.with_xor_complement(edge.complemented),
                node_by_id,
                builder,
                cache,
            );
            builder.branch(imported.variable_index, low, high)
        }
    };

    cache.insert(edge, node_id);
    node_id
}

#[cfg(test)]
mod tests {
    use super::{Bdd, ImportedBdd, ImportedBranchNode, ImportedEdge};

    #[test]
    fn reduces_redundant_subgraphs() {
        let bdd = Bdd::from_truth_table(3, &[0, 0, 1, 1, 0, 0, 1, 1]);
        assert!(bdd.node_count() < 8);
    }

    #[test]
    fn evaluates_truth_table_entries() {
        let truth_table = [0, 1, 1, 0, 1, 0, 0, 1];
        let bdd = Bdd::from_truth_table(3, &truth_table);

        for (assignment, expected) in truth_table.into_iter().enumerate() {
            assert_eq!(bdd.evaluate(assignment), expected);
        }
    }

    #[test]
    fn imports_cudd_style_complemented_root() {
        let imported = ImportedBdd {
            num_vars: 1,
            root: ImportedEdge::complemented_node(7),
            nodes: vec![ImportedBranchNode {
                id: 7,
                variable_index: 0,
                low: ImportedEdge::terminal_zero(),
                high: ImportedEdge::terminal_one(),
            }],
        };

        let bdd = Bdd::from_imported_boolean_dag(&imported);
        assert_eq!(bdd.evaluate(0), 1);
        assert_eq!(bdd.evaluate(1), 0);
    }

    #[test]
    fn imports_shared_subgraph() {
        let imported = ImportedBdd {
            num_vars: 3,
            root: ImportedEdge::node(10),
            nodes: vec![
                ImportedBranchNode {
                    id: 2,
                    variable_index: 2,
                    low: ImportedEdge::terminal_zero(),
                    high: ImportedEdge::terminal_one(),
                },
                ImportedBranchNode {
                    id: 10,
                    variable_index: 0,
                    low: ImportedEdge::node(2),
                    high: ImportedEdge::node(2),
                },
            ],
        };

        let bdd = Bdd::from_imported_boolean_dag(&imported);
        assert_eq!(bdd.node_count(), 3);
        assert_eq!(bdd.evaluate(0b000), 0);
        assert_eq!(bdd.evaluate(0b100), 1);
    }
}
