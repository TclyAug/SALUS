use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use salus::{
    build_merge_lut2_groups, from_blif_circuit_with_dags, reduce_lut_with_cudd,
    write_packed_groups_to_dir, BlifCircuit, FUSED_SELECTOR_GROUP_BITS,
};

fn main() -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let input_path = PathBuf::from(
        args.first()
            .filter(|arg| !arg.starts_with("--"))
            .cloned()
            .ok_or(
                "usage: generate_circuit <input.{v|blif}> <output_dir> [--abc-bin PATH] [--skip-abc] [--max-abc-iters N] [--with-merge]",
            )?,
    );
    if !args.is_empty() && !args[0].starts_with("--") {
        args.remove(0);
    }
    let output_dir = PathBuf::from(
        args.first()
            .filter(|arg| !arg.starts_with("--"))
            .cloned()
            .ok_or(
                "usage: generate_circuit <input.{v|blif}> <output_dir> [--abc-bin PATH] [--skip-abc] [--max-abc-iters N] [--with-merge]",
            )?,
    );
    if !args.is_empty() && !args[0].starts_with("--") {
        args.remove(0);
    }

    let abc_bin = take_option(&mut args, "--abc-bin").unwrap_or_else(default_abc_bin);
    let skip_abc = take_flag(&mut args, "--skip-abc");
    let with_merge = take_flag(&mut args, "--with-merge");
    let max_abc_iters = if let Some(raw) = take_option(&mut args, "--max-abc-iters") {
        raw.parse::<usize>()?
    } else {
        8
    };
    if max_abc_iters == 0 {
        return Err("--max-abc-iters must be at least 1".into());
    }
    if !args.is_empty() {
        return Err(format!("unknown arguments: {}", args.join(" ")).into());
    }

    fs::create_dir_all(&output_dir)?;
    let mapped_blif = if skip_abc {
        if input_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("blif"))
            != Some(true)
        {
            return Err("--skip-abc requires a BLIF input".into());
        }
        let mapped_path = output_dir.join("mapped_lut15.blif");
        if input_path != mapped_path {
            fs::copy(&input_path, &mapped_path)?;
        }
        mapped_path
    } else {
        map_to_lut15_until_stable(&input_path, &output_dir, &abc_bin, max_abc_iters)?
    };

    let circuit = BlifCircuit::parse_file(&mapped_blif)?;
    let dags_dir = output_dir.join("dags");
    fs::create_dir_all(&dags_dir)?;
    let tool_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tools")
        .join("cudd_tt_to_dag");

    let mut dag_paths = Vec::with_capacity(circuit.luts.len());
    for (idx, lut) in circuit.luts.iter().enumerate() {
        let imported = reduce_lut_with_cudd(&tool_path, lut)?;
        let file_name = format!("{idx:06}_{}.dag", lut.name);
        let rel_path = PathBuf::from("dags").join(&file_name);
        let dag_path = output_dir.join(&rel_path);
        write_imported_dag(&dag_path, &imported)?;
        dag_paths.push(rel_path);
    }

    let preprocessed = from_blif_circuit_with_dags(&circuit, &dag_paths);
    preprocessed.write_to_dir(&output_dir)?;

    if with_merge {
        let packed_groups = build_merge_lut2_groups(&circuit, 15, FUSED_SELECTOR_GROUP_BITS);
        write_packed_groups_to_dir(&output_dir, &packed_groups)?;
    } else {
        let packed_groups_path = output_dir.join("packed_groups.txt");
        if packed_groups_path.exists() {
            fs::remove_file(packed_groups_path)?;
        }
    }

    println!("output_dir: {}", output_dir.display());
    println!("generation_ms: {:.2}", started.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}

fn map_to_lut15_until_stable(
    input_path: &Path,
    output_dir: &Path,
    abc_bin: &str,
    max_iters: usize,
) -> Result<PathBuf, Box<dyn Error>> {
    let ext = input_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if ext != "blif" && ext != "v" && ext != "sv" && ext != "verilog" {
        return Err(format!(
            "unsupported input format for {}: expected .blif, .v, .sv, or .verilog",
            input_path.display()
        )
        .into());
    }

    let abc_iters_dir = output_dir.join("abc_iters");
    fs::create_dir_all(&abc_iters_dir)?;

    let mut current_input = input_path.to_path_buf();
    let mut best_lut_count = usize::MAX;
    let mut best_output = None;

    for iter in 0..max_iters {
        let candidate_output = abc_iters_dir.join(format!("iter_{:02}.blif", iter + 1));
        run_abc_lut15_pass(
            abc_bin,
            &current_input,
            &candidate_output,
            iter == 0 && ext != "blif",
        )?;
        let candidate_circuit = BlifCircuit::parse_file(&candidate_output)?;
        let candidate_lut_count = candidate_circuit.luts.len();

        if candidate_lut_count < best_lut_count {
            best_lut_count = candidate_lut_count;
            best_output = Some(candidate_output.clone());
            current_input = candidate_output;
        } else {
            break;
        }
    }

    let best_output = best_output.ok_or("ABC did not produce a mapped BLIF")?;
    let mapped_path = output_dir.join("mapped_lut15.blif");
    if best_output != mapped_path {
        fs::copy(best_output, &mapped_path)?;
    }
    Ok(mapped_path)
}

fn run_abc_lut15_pass(
    abc_bin: &str,
    input_path: &Path,
    output_path: &Path,
    is_verilog: bool,
) -> Result<(), Box<dyn Error>> {
    let read_cmd = if is_verilog {
        format!("read_verilog {}", shell_escape(input_path))
    } else {
        format!("read_blif {}", shell_escape(input_path))
    };
    let script = format!(
        "{read_cmd}; strash; dch; if -K 15; write_blif {}",
        shell_escape(output_path),
    );

    let output = Command::new(abc_bin)
        .arg("-q")
        .arg(script)
        .output()
        .map_err(|err| {
            format!(
                "failed to launch ABC via '{}': {err}; install abc or pass --abc-bin",
                abc_bin
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "ABC failed on {}: {}",
            input_path.display(),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(())
}

fn write_imported_dag(
    dag_path: &Path,
    imported: &salus::ImportedDagData,
) -> Result<(), Box<dyn Error>> {
    let mut text = String::new();
    text.push_str(&format!("num_vars {}\n", imported.imported_bdd.num_vars));
    text.push_str(&format!(
        "truth_table {}\n",
        imported
            .truth_table
            .iter()
            .map(|bit| if *bit == 0 { '0' } else { '1' })
            .collect::<String>()
    ));
    text.push_str(&format!(
        "root {} {}\n",
        imported
            .imported_bdd
            .root
            .node_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "-1".to_string()),
        if imported.imported_bdd.root.complemented {
            1
        } else {
            0
        }
    ));
    for node in &imported.imported_bdd.nodes {
        text.push_str(&format!(
            "node {} {} {} {} {} {}\n",
            node.id,
            node.variable_index,
            node.low
                .node_id
                .map(|id| id.to_string())
                .unwrap_or_else(|| "-1".to_string()),
            if node.low.complemented { 1 } else { 0 },
            node.high
                .node_id
                .map(|id| id.to_string())
                .unwrap_or_else(|| "-1".to_string()),
            if node.high.complemented { 1 } else { 0 }
        ));
    }
    fs::write(dag_path, text)?;
    Ok(())
}

fn shell_escape(path: &Path) -> String {
    let raw = path.display().to_string();
    format!("'{}'", raw.replace('\'', "'\"'\"'"))
}

fn default_abc_bin() -> String {
    for candidate in default_abc_candidates() {
        if candidate.is_file() {
            return candidate.display().to_string();
        }
    }
    "abc".to_string()
}

fn default_abc_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    candidates.push(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("third_party")
            .join("abc")
            .join("abc"),
    );

    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(repo_root) = current_exe
            .parent()
            .and_then(|path| path.parent())
            .and_then(|path| path.parent())
        {
            candidates.push(repo_root.join("third_party").join("abc").join("abc"));
        }
    }

    if let Ok(current_dir) = std::env::current_dir() {
        candidates.push(current_dir.join("third_party").join("abc").join("abc"));
    }

    candidates
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
    let index = args.iter().position(|arg| arg == flag)?;
    args.remove(index);
    if index < args.len() {
        Some(args.remove(index))
    } else {
        None
    }
}
