use std::time::Duration;

#[derive(Clone, Debug, Default)]
pub struct TimedStat {
    pub count: u64,
    pub total_ns: u128,
    pub total_sq_ns: f64,
}

impl TimedStat {
    pub fn record(&mut self, duration: Duration) {
        let ns = duration.as_nanos();
        self.count += 1;
        self.total_ns += ns;
        self.total_sq_ns += (ns as f64) * (ns as f64);
    }

    pub fn total_ms(&self) -> f64 {
        self.total_ns as f64 / 1_000_000.0
    }

    pub fn mean_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_ms() / self.count as f64
        }
    }

    pub fn variance_ms2(&self) -> f64 {
        if self.count <= 1 {
            0.0
        } else {
            let count = self.count as f64;
            let mean_ns = self.total_ns as f64 / count;
            let variance_ns2 = (self.total_sq_ns / count) - mean_ns * mean_ns;
            variance_ns2.max(0.0) / 1_000_000_000_000.0
        }
    }

    pub fn merge(&mut self, other: &TimedStat) {
        self.count += other.count;
        self.total_ns += other.total_ns;
        self.total_sq_ns += other.total_sq_ns;
    }
}

#[derive(Clone, Debug, Default)]
pub struct ComponentTimingStats {
    pub input_scalar_bootstrap: TimedStat,
    pub input_group_bootstrap: TimedStat,
    pub sample_extract: TimedStat,
    pub key_switch: TimedStat,
    pub blind_rotate: TimedStat,
    pub conversion: TimedStat,
    pub fourier_conversion: TimedStat,
    pub cmux: TimedStat,
    pub bdd_tree: TimedStat,
    pub packed_group_eval: TimedStat,
    pub packed_group_refresh: TimedStat,
    pub grouped_lut_eval: TimedStat,
    pub grouped_refresh: TimedStat,
    pub singleton_lut_eval: TimedStat,
    pub singleton_refresh: TimedStat,
    pub total_circuit_eval: TimedStat,
    pub bdd_nodes: u64,
    pub packed_group_eval_outputs: u64,
    pub packed_group_refresh_outputs: u64,
    pub grouped_refresh_outputs: u64,
}

impl ComponentTimingStats {
    pub fn merge(&mut self, other: &ComponentTimingStats) {
        self.input_scalar_bootstrap
            .merge(&other.input_scalar_bootstrap);
        self.input_group_bootstrap
            .merge(&other.input_group_bootstrap);
        self.sample_extract.merge(&other.sample_extract);
        self.key_switch.merge(&other.key_switch);
        self.blind_rotate.merge(&other.blind_rotate);
        self.conversion.merge(&other.conversion);
        self.fourier_conversion.merge(&other.fourier_conversion);
        self.cmux.merge(&other.cmux);
        self.bdd_tree.merge(&other.bdd_tree);
        self.packed_group_eval.merge(&other.packed_group_eval);
        self.packed_group_refresh.merge(&other.packed_group_refresh);
        self.grouped_lut_eval.merge(&other.grouped_lut_eval);
        self.grouped_refresh.merge(&other.grouped_refresh);
        self.singleton_lut_eval.merge(&other.singleton_lut_eval);
        self.singleton_refresh.merge(&other.singleton_refresh);
        self.total_circuit_eval.merge(&other.total_circuit_eval);
        self.bdd_nodes += other.bdd_nodes;
        self.packed_group_eval_outputs += other.packed_group_eval_outputs;
        self.packed_group_refresh_outputs += other.packed_group_refresh_outputs;
        self.grouped_refresh_outputs += other.grouped_refresh_outputs;
    }

    pub fn record_input_scalar_bootstrap(&mut self, duration: Duration) {
        self.input_scalar_bootstrap.record(duration);
    }

    pub fn record_input_group_bootstrap(&mut self, duration: Duration) {
        self.input_group_bootstrap.record(duration);
    }

    pub fn record_sample_extract(&mut self, duration: Duration) {
        self.sample_extract.record(duration);
    }

    pub fn record_key_switch(&mut self, duration: Duration) {
        self.key_switch.record(duration);
    }

    pub fn record_blind_rotate(&mut self, duration: Duration) {
        self.blind_rotate.record(duration);
    }

    pub fn record_conversion(&mut self, duration: Duration) {
        self.conversion.record(duration);
    }

    pub fn record_fourier_conversion(&mut self, duration: Duration) {
        self.fourier_conversion.record(duration);
    }

    pub fn record_cmux(&mut self, duration: Duration) {
        self.cmux.record(duration);
    }

    pub fn record_bdd_tree(&mut self, duration: Duration, branch_nodes: usize) {
        self.bdd_tree.record(duration);
        self.bdd_nodes += branch_nodes as u64;
    }

    pub fn record_packed_group_eval(&mut self, duration: Duration, output_count: usize) {
        self.packed_group_eval.record(duration);
        self.packed_group_eval_outputs += output_count as u64;
    }

    pub fn record_packed_group_refresh(&mut self, duration: Duration, output_count: usize) {
        self.packed_group_refresh.record(duration);
        self.packed_group_refresh_outputs += output_count as u64;
    }

    pub fn record_grouped_lut_eval(&mut self, duration: Duration) {
        self.grouped_lut_eval.record(duration);
    }

    pub fn record_grouped_refresh(&mut self, duration: Duration, output_count: usize) {
        self.grouped_refresh.record(duration);
        self.grouped_refresh_outputs += output_count as u64;
    }

    pub fn record_singleton_lut_eval(&mut self, duration: Duration) {
        self.singleton_lut_eval.record(duration);
    }

    pub fn record_singleton_refresh(&mut self, duration: Duration) {
        self.singleton_refresh.record(duration);
    }

    pub fn record_total_circuit_eval(&mut self, duration: Duration) {
        self.total_circuit_eval.record(duration);
    }
}
