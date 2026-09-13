//! CPU comparison of every preregistered trajectory pair/checkpoint.
use super::*;
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Comparison {
    schema_version: String,
    teacher: PathBuf,
    binary_sha256: String,
    output: PathBuf,
    pairs: Vec<Pair>,
    provenance: Value,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Pair {
    id: String,
    a: PathBuf,
    b: PathBuf,
    a_registration: PathBuf,
    b_registration: PathBuf,
    a_registration_sha256: String,
    b_registration_sha256: String,
}
fn checkpoint(dir: &Path, n: usize, registration: &str) -> Result<Value, String> {
    let row = issue16::read_json(&dir.join(n.to_string()).join("metadata.json"))?;
    if row["registration_sha256"] != registration
        || row["step"] != n
        || row["clean_completion"] != true
        || row["state_sha256"] != file_hash(&dir.join(n.to_string()).join("state.bin"))?
    {
        return Err("DATA_INTEGRITY_STOP: comparison checkpoint differs".into());
    }
    validate_resume_identity(&row["resume"], n, 4)?;
    Ok(row)
}
pub fn run(path: &Path) -> Result<(), String> {
    let s: Comparison = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| format!("CONFIGURATION_STOP: {e}"))?;
    if s.schema_version != "issue16-comparison-v1"
        || s.pairs.is_empty()
        || s.pairs.len() > 16
        || file_hash(&std::env::current_exe().map_err(|e| e.to_string())?)? != s.binary_sha256
    {
        return Err("CONFIGURATION_STOP: comparison contract".into());
    }
    let registration = file_hash(path)?;
    let args = issue16::fixed_args(&s.teacher)?;
    fs::create_dir_all(&s.output).map_err(|e| format!("RESOURCE_STOP: {e}"))?;
    let mut results = Vec::new();
    for pair in &s.pairs {
        if pair.id.is_empty() || pair.id.chars().any(|c| !c.is_ascii_alphanumeric() && c != '-') {
            return Err("CONFIGURATION_STOP: invalid pair id".into());
        }
        if file_hash(&pair.a_registration)? != pair.a_registration_sha256
            || file_hash(&pair.b_registration)? != pair.b_registration_sha256
        {
            return Err("DATA_INTEGRITY_STOP: source registration changed".into());
        }
        for n in [10, 100, 200, 400, 700, 1000] {
            let a = checkpoint(&pair.a, n, &pair.a_registration_sha256)?;
            let b = checkpoint(&pair.b, n, &pair.b_registration_sha256)?;
            if a["initial"] != b["initial"] || a["consumed_batches"] != b["consumed_batches"] {
                return Err("DATA_INTEGRITY_STOP: pair input/history differs".into());
            }
            let output = s.output.join(format!("{}-{n}.json", pair.id));
            if output.exists() {
                let row = issue16::read_json(&output)?;
                if row["registration_sha256"] != registration
                    || row["a_state_sha256"] != a["state_sha256"]
                    || row["b_state_sha256"] != b["state_sha256"]
                {
                    return Err("DATA_INTEGRITY_STOP: resumed comparison differs".into());
                }
                results.push(row);
                continue;
            }
            eprintln!("issue16 compare pair={} step={n}", pair.id);
            let al = crate::load_cuda_cpp_sfnn_initial_state(
                &pair.a.join(n.to_string()).join("state.bin"),
                &args,
                crate::CudaCppSfnnFeatureKind::Halfka2,
            )?;
            let bl = crate::load_cuda_cpp_sfnn_initial_state(
                &pair.b.join(n.to_string()).join("state.bin"),
                &args,
                crate::CudaCppSfnnFeatureKind::Halfka2,
            )?;
            if al.completed_steps != n || bl.completed_steps != n || al.optimizer_steps != n || bl.optimizer_steps != n
            {
                return Err("DATA_INTEGRITY_STOP: loaded steps differ".into());
            }
            let aw = crate::sfnn_initial_weights_into_readback(al.weights);
            let bw = crate::sfnn_initial_weights_into_readback(bl.weights);
            let parameters = compare_tolerance(weights_map(&aw), weights_map(&bw), 1e-5, 1e-3)?;
            let optimizer = compare_tolerance(
                optimizer_host_map(al.optimizer_states.as_ref().ok_or("missing optimizer")?.as_host()),
                optimizer_host_map(bl.optimizer_states.as_ref().ok_or("missing optimizer")?.as_host()),
                1e-5,
                1e-3,
            )?;
            let floats = |v: &Value| -> Result<Vec<f32>, String> {
                v.as_array()
                    .ok_or("DATA_INTEGRITY_STOP: invalid outputs")?
                    .iter()
                    .map(|x| x.as_f64().map(|x| x as f32).ok_or("DATA_INTEGRITY_STOP: invalid float".into()))
                    .collect()
            };
            let f =
                metric_tolerance(&floats(&a["float_outputs"])?, &floats(&b["float_outputs"])?, vec![16], 1e-5, 1e-3)?;
            let ft = (a["quantized"]["ft_weight_nonzero_fraction"].as_f64().ok_or("missing FT")?
                - b["quantized"]["ft_weight_nonzero_fraction"].as_f64().ok_or("missing FT")?)
            .abs();
            let within = all_passed(&parameters) && all_passed(&optimizer) && f["passed"] == true && ft <= 1e-5;
            let row = json!({"registration_sha256":registration,"pair":pair.id,"step":n,"a_state_sha256":a["state_sha256"],"b_state_sha256":b["state_sha256"],"parameters":parameters,"optimizer":optimizer,"float_output":f,"ft_nonzero_abs_delta":ft,"within_original_gate1_tolerance":within});
            publish(&output, &row)?;
            results.push(row);
        }
    }
    publish(
        &s.output.join("REPORT.json"),
        &json!({"schema_version":"issue16-comparison-report-v1","registration_sha256":registration,"completed":true,"comparisons":results,"qualification_passed":false,"issue12_decision":"NO_GO_PHASE2"}),
    )?;
    Ok(())
}
