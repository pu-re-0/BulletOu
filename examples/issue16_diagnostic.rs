//! Explicit diagnostic imports preserve ordinary qualification/cross-resume contracts.
use super::*;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    arm: String,
    state: PathBuf,
    state_sha256: String,
    metadata: PathBuf,
    metadata_sha256: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Diagnostic {
    schema_version: String,
    teacher: PathBuf,
    teacher_sha256: String,
    binary_sha256: String,
    output: PathBuf,
    sources: Vec<Source>,
    repetitions: usize,
    wall_seconds: u64,
    sealed_test: bool,
    provenance: Value,
}
impl Diagnostic {
    fn source_step(&self) -> usize {
        if self.schema_version == "issue16-interval-update-v1" { 200 } else { 100 }
    }
}
// Dataset identities belong to the caller's immutable registration, not source.
pub(super) fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn validate(s: &Diagnostic) -> Result<(), String> {
    if s.sealed_test {
        return Err("SEALED_POLICY_STOP: diagnostic input".into());
    }
    if !matches!(
        (s.schema_version.as_str(), s.wall_seconds),
        ("issue16-one-update-v1", 1800) | ("issue16-interval-update-v1", 2400)
    ) || s.repetitions != 2
        || s.sources.len() != 2
        || s.sources[0].arm != "D-A"
        || s.sources[1].arm != "D-A-prime"
        || !valid_sha256(&s.teacher_sha256)
    {
        return Err("CONFIGURATION_STOP: fixed Issue16 contract differs".into());
    }
    Ok(())
}
pub(super) fn read_json(p: &Path) -> Result<Value, String> {
    serde_json::from_slice(&fs::read(p).map_err(|e| format!("DATA_INTEGRITY_STOP: {e}"))?)
        .map_err(|e| format!("DATA_INTEGRITY_STOP: {e}"))
}
fn source_metadata(s: &Source, seq: &[Batch], completed_steps: usize) -> Result<Value, String> {
    if file_hash(&s.state)? != s.state_sha256 || file_hash(&s.metadata)? != s.metadata_sha256 {
        return Err("DATA_INTEGRITY_STOP: source hash differs".into());
    }
    let m = read_json(&s.metadata)?;
    validate_resume_identity(&m["resume"], completed_steps, 4)?;
    let consumed: Vec<_> =
        (1..=completed_steps).map(|n| json!({"step":n,"batch_sha256":digest_json(&seq[(n-1)%4])})).collect();
    if m["arm"] != s.arm
        || m["step"] != completed_steps
        || m["state_sha256"] != s.state_sha256
        || m["consumed_batches"] != json!(consumed)
        || m["clean_completion"] != true
    {
        return Err("DATA_INTEGRITY_STOP: source identity differs".into());
    }
    Ok(m)
}
pub(super) fn fixed_args(teacher: &Path) -> Result<crate::Args, String> {
    crate::Args::try_parse_from([
        "bulletou",
        "--arch",
        "SFNN_halfka2_1024_7_64_k3k3",
        "--backend",
        "cuda-cpp",
        "--teacher",
        teacher.to_str().ok_or("invalid path")?,
        "--cuda-cpp-train-steps",
        "1",
        "--sfnn-factorizer",
        "shared",
    ])
    .map_err(|e| format!("CONFIGURATION_STOP: {e}"))
}
fn sequence(s: &Diagnostic) -> Result<Vec<Batch>, String> {
    batches(&Spec {
        schema_version: String::new(),
        stage: String::new(),
        teacher: s.teacher.clone(),
        teacher_sha256: s.teacher_sha256.clone(),
        binary_sha256: s.binary_sha256.clone(),
        output: s.output.clone(),
        seed: SEED,
        repetitions: 2,
        absolute_tolerance: ATOL,
        relative_tolerance: RTOL,
        fixtures: vec![],
        sealed_test: false,
        provenance: Value::Null,
        gate0_report: None,
        gate0_report_sha256: None,
    })
}
fn import(
    ctx: &Context,
    args: &crate::Args,
    source: &Source,
    b: &Batch,
    selector: SfnnL0BackwardSelector,
    completed_steps: usize,
) -> Result<SfnnTrainStepRunner, String> {
    let loaded = crate::load_cuda_cpp_sfnn_initial_state(&source.state, args, crate::CudaCppSfnnFeatureKind::Halfka2)?;
    if loaded.completed_steps != completed_steps || loaded.optimizer_steps != completed_steps {
        return Err("DATA_INTEGRITY_STOP: binary state steps differ".into());
    }
    let mut r = runner(ctx, loaded.weights.as_host(), b, selector)?;
    r.upload_state_from_host(
        ctx,
        loaded.weights.as_host(),
        loaded.optimizer_states.as_ref().ok_or("DATA_INTEGRITY_STOP: missing optimizer")?.as_host(),
        SfnnFactorizerActive { shared: true, ..SfnnFactorizerActive::NONE },
        SfnnFactorizerAlpha::ONE,
    )
    .map_err(|e| e.to_string())?;
    let expected = json!({"weights":hash_tensors(&weights_map(&crate::sfnn_initial_weights_into_readback(loaded.weights)))?,"optimizer":hash_tensors(&optimizer_host_map(loaded.optimizer_states.as_ref().unwrap().as_host()))?});
    if state_identity(ctx, &r)? != expected {
        return Err("IMPLEMENTATION_STOP: complete state upload changed".into());
    }
    Ok(r)
}
pub(super) fn state_identity(ctx: &Context, r: &SfnnTrainStepRunner) -> Result<Value, String> {
    Ok(
        json!({"weights":hash_tensors(&weights_map(&r.read_weights(ctx).map_err(|e|e.to_string())?))?,"optimizer":hash_tensors(&optimizer_map(&r.read_optimizer_states(ctx).map_err(|e|e.to_string())?))?}),
    )
}
fn execute_pair(
    s: &Diagnostic,
    args: &crate::Args,
    seq: &[Batch],
    source: &Source,
    rep: usize,
) -> Result<Value, String> {
    let ctx = Context::new(0).map_err(|e| format!("RESOURCE_STOP: {e}"))?;
    let update = s.source_step() + 1;
    let b = &seq[s.source_step() % seq.len()];
    let mut traces = Vec::new();
    let mut initial = None;
    let mut observations = Vec::new();
    for selector in [SfnnL0BackwardSelector::InverseIndex, SfnnL0BackwardSelector::MaterializedSparse] {
        let mut r = import(&ctx, args, source, b, selector, s.source_step())?;
        let before = state_identity(&ctx, &r)?;
        if initial.as_ref().is_some_and(|v| v != &before) {
            return Err("IMPLEMENTATION_STOP: unequal imported state".into());
        }
        initial = Some(before.clone());
        step(&ctx, &mut r, b, "ordinary", false, update)?;
        let f = r.read_forward_trace(&ctx).map_err(|e| e.to_string())?;
        let l = r.read_loss(&ctx).map_err(|e| e.to_string())?;
        let g = r.read_backward_trace(&ctx).map_err(|e| e.to_string())?;
        let unchanged = state_identity(&ctx, &r)? == before;
        let reference = l0_reference(&f, &g, b)?;
        observations.push(json!({"resolved_path":format!("{:?}",r.l0_backward_path()),"backward_did_not_update_state":unchanged,"cpu_f64_l0":reference}));
        traces.push((f, l, g));
    }
    let forward = compare(forward_map(&traces[0].0, &traces[0].1), forward_map(&traces[1].0, &traces[1].1))?;
    let gradients = compare(gradient_map(&traces[0].2), gradient_map(&traces[1].2))?;
    drop(traces);
    let mut updates = Vec::new();
    let mut state_hashes = Vec::new();
    for (arm, selector) in
        [("D-A", SfnnL0BackwardSelector::InverseIndex), ("D-A-prime", SfnnL0BackwardSelector::MaterializedSparse)]
    {
        let mut r = import(&ctx, args, source, b, selector, s.source_step())?;
        if Some(state_identity(&ctx, &r)?) != initial {
            return Err("IMPLEMENTATION_STOP: observation changed source".into());
        }
        step(&ctx, &mut r, b, "ordinary", true, update)?;
        let w = r.read_weights(&ctx).map_err(|e| e.to_string())?;
        let o = r.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
        let path = s.output.join(format!("{}-{rep}-{arm}-{}-{update}.bin", source.arm, std::process::id()));
        if path.exists() || path.with_extension("bin.tmp").exists() {
            return Err("DATA_INTEGRITY_STOP: preserve incomplete state; use a new attempt".into());
        }
        crate::write_cuda_cpp_sfnn_weights_bin(&path, &w, PostPairwiseTransform::Identity, &o, None, update, update)?;
        state_hashes.push(json!({"arm":arm,"state_sha256":file_hash(&path)?,"state":path}));
        updates.push((w, o));
    }
    let parameters = compare(weights_map(&updates[0].0), weights_map(&updates[1].0))?;
    let optimizer = compare(optimizer_map(&updates[0].1), optimizer_map(&updates[1].1))?;
    let passed = all_passed(&forward)
        && all_passed(&gradients)
        && all_passed(&parameters)
        && all_passed(&optimizer)
        && observations.iter().all(|v| v["backward_did_not_update_state"] == true && all_passed(&v["cpu_f64_l0"]));
    if file_hash(&source.state)? != source.state_sha256 {
        return Err("DATA_INTEGRITY_STOP: source mutated".into());
    }
    Ok(
        json!({"source_arm":source.arm,"source_state_sha256":source.state_sha256,"repetition":rep,"update":update,"batch_sha256":digest_json(b),"initial":initial,"observations":observations,"forward":forward,"gradients":gradients,"parameters":parameters,"optimizer":optimizer,"updated_states":state_hashes,"passed":passed}),
    )
}
pub fn run(path: &Path) -> Result<(), String> {
    if read_json(path)?["schema_version"] == "issue16-comparison-v1" {
        return super::issue16_compare::run(path);
    }
    if read_json(path)?["schema_version"] == "issue16-trajectory-v1" {
        return super::issue16_trajectory::run(path);
    }
    let s: Diagnostic = serde_json::from_slice(&fs::read(path).map_err(|e| format!("DATA_INTEGRITY_STOP: {e}"))?)
        .map_err(|e| format!("CONFIGURATION_STOP: {e}"))?;
    validate(&s)?;
    if file_hash(&s.teacher)? != s.teacher_sha256
        || file_hash(&std::env::current_exe().map_err(|e| e.to_string())?)? != s.binary_sha256
    {
        return Err("DATA_INTEGRITY_STOP: input/binary mismatch".into());
    }
    let args = fixed_args(&s.teacher)?;
    let seq = sequence(&s)?;
    for source in &s.sources {
        source_metadata(source, &seq, s.source_step())?;
    }
    let registration = file_hash(path)?;
    if s.output.exists() {
        if read_json(&s.output.join("DISPATCH.json"))?["registration_sha256"] != registration
            || s.output.join("REPORT.json").exists()
        {
            return Err("DATA_INTEGRITY_STOP: output already completed or registration changed".into());
        }
    } else {
        fs::create_dir_all(&s.output).map_err(|e| format!("RESOURCE_STOP: {e}"))?;
        publish(&s.output.join("DISPATCH.json"), &json!({"registration_sha256":registration,"spec":s}))?;
    }
    let start = Instant::now();
    let mut rows = Vec::new();
    for source in &s.sources {
        for rep in 0..s.repetitions {
            if start.elapsed().as_secs() >= s.wall_seconds {
                return Err("RESOURCE_STOP: diagnostic wall budget".into());
            }
            let out = s.output.join(format!("{}-{rep}.json", source.arm));
            let row = if out.exists() {
                let row = read_json(&out)?;
                if row["registration_sha256"] != registration {
                    return Err("DATA_INTEGRITY_STOP: resumed row differs".into());
                }
                if row["source_arm"] != source.arm
                    || row["source_state_sha256"] != source.state_sha256
                    || row["repetition"] != rep
                    || row["update"] != s.source_step() + 1
                    || row["batch_sha256"] != digest_json(&seq[s.source_step() % seq.len()])
                {
                    return Err("DATA_INTEGRITY_STOP: resumed identity differs".into());
                }
                let states = row["updated_states"].as_array().ok_or("DATA_INTEGRITY_STOP: missing updated states")?;
                if states.len() != 2 {
                    return Err("DATA_INTEGRITY_STOP: incomplete updated states".into());
                }
                for state in states {
                    if file_hash(Path::new(state["state"].as_str().ok_or("DATA_INTEGRITY_STOP: invalid state path")?))?
                        != state["state_sha256"]
                    {
                        return Err("DATA_INTEGRITY_STOP: updated state changed".into());
                    }
                }
                row
            } else {
                eprintln!("issue16 source={} repetition={rep} update={}", source.arm, s.source_step() + 1);
                let mut row = match execute_pair(&s, &args, &seq, source, rep) {
                    Ok(row) => row,
                    Err(error) => {
                        publish(
                            &s.output.join(format!("STOP-{}.json", std::process::id())),
                            &json!({"registration_sha256":registration,"source_arm":source.arm,"repetition":rep,"typed_stop":stop_kind(&error),"error":error,"completed":false}),
                        )?;
                        return Err(error);
                    }
                };
                row["registration_sha256"] = json!(registration);
                publish(&out, &row)?;
                row
            };
            rows.push(row);
        }
    }
    let passed = rows.iter().all(|v| v["passed"] == true);
    publish(
        &s.output.join("REPORT.json"),
        &json!({"schema_version":if s.source_step() == 100 {"issue16-one-update-report-v1"} else {"issue16-interval-update-report-v1"},"build_info":crate::bulletou_build_info_text(),"registration_sha256":registration,"binary_sha256":s.binary_sha256,"rows":rows,"passed":passed,"typed_stop":if passed{Value::Null}else{json!("IMPLEMENTATION_STOP")},"elapsed_seconds":start.elapsed().as_secs_f64(),"issue13_accepted":false,"issue12_decision":"NO_GO_PHASE2"}),
    )?;
    if !passed {
        return Err("IMPLEMENTATION_STOP: one-update comparison; see REPORT.json".into());
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contract_rejects_sealed_and_changed_repetitions() {
        let mut s:Diagnostic=serde_json::from_value(json!({"schema_version":"issue16-one-update-v1","teacher":"unused","teacher_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","binary_sha256":"unused","output":"unused","sources":[{"arm":"D-A","state":"a","state_sha256":"a","metadata":"a","metadata_sha256":"a"},{"arm":"D-A-prime","state":"b","state_sha256":"b","metadata":"b","metadata_sha256":"b"}],"repetitions":2,"wall_seconds":1800,"sealed_test":false,"provenance":{}})).unwrap();
        assert!(validate(&s).is_ok());
        assert_eq!(s.source_step(), 100);
        s.schema_version = "issue16-interval-update-v1".into();
        assert!(validate(&s).is_err());
        s.wall_seconds = 2400;
        assert!(validate(&s).is_ok());
        assert_eq!(s.source_step(), 200);
        s.schema_version = "issue16-unknown-v1".into();
        assert!(validate(&s).is_err());
        s.schema_version = "issue16-interval-update-v1".into();
        s.repetitions = 1;
        assert!(validate(&s).is_err());
        s.sealed_test = true;
        assert!(validate(&s).unwrap_err().starts_with("SEALED_POLICY_STOP"));
    }
    #[test]
    fn diagnostic_cli_rejects_ordinary_resume_and_selector() {
        for extra in
            [vec!["--resume"], vec!["--initial-state", "unused"], vec!["--sfnn-l0-backward", "materialized-sparse"]]
        {
            let mut cli = vec![
                "bulletou",
                "--arch",
                "SFNN_halfka2_1024_7_64_k3k3",
                "--backend",
                "cuda-cpp",
                "--teacher",
                "unused",
                "--cuda-cpp-train-steps",
                "1",
                "--issue16-diagnostic",
                "unused",
            ];
            cli.extend(extra);
            let args = crate::Args::try_parse_from(cli).unwrap();
            assert!(args.validate_cuda_cpp_backend_options().is_err());
        }
    }
    #[test]
    fn digest_format_is_strict() {
        assert!(valid_sha256(&"a".repeat(64)));
        for value in ["".to_string(), "a".repeat(63), "g".repeat(64), "A".repeat(64)] {
            assert!(!valid_sha256(&value));
        }
    }
    #[test]
    fn interval_metadata_requires_the_actual_source_step_and_history() {
        let dir = std::env::temp_dir().join(format!("issue16-interval-meta-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let state = dir.join("state.bin");
        let metadata = dir.join("metadata.json");
        fs::write(&state, b"small host-only fixture").unwrap();
        let seq: Vec<_> = (0..4)
            .map(|n| Batch {
                stm: vec![],
                nstm: vec![],
                buckets: vec![n],
                targets: vec![0.5],
                weights: vec![1.0],
                max_active: 0,
            })
            .collect();
        let hash = file_hash(&state).unwrap();
        let mut row = json!({"arm":"D-A", "step":200, "state_sha256":hash,
            "clean_completion":true, "resume":resume_identity(200, 4),
            "consumed_batches":(1..=200).map(|n| json!({"step":n,"batch_sha256":digest_json(&seq[(n-1)%4])})).collect::<Vec<_>>()});
        fs::write(&metadata, serde_json::to_vec(&row).unwrap()).unwrap();
        let mut source = Source {
            arm: "D-A".into(),
            state,
            state_sha256: hash,
            metadata: metadata.clone(),
            metadata_sha256: file_hash(&metadata).unwrap(),
        };
        assert!(source_metadata(&source, &seq, 200).is_ok());
        assert!(source_metadata(&source, &seq, 100).is_err());
        row["resume"]["optimizer_steps"] = json!(201);
        fs::write(&metadata, serde_json::to_vec(&row).unwrap()).unwrap();
        source.metadata_sha256 = file_hash(&metadata).unwrap();
        assert!(source_metadata(&source, &seq, 200).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn update_number_preserves_ranger_phase() {
        assert_eq!(parameters(101).radam.step, 101);
        assert_eq!(parameters(201).radam.step, 201);
        assert_eq!(parameters(102).lookahead_period, 6);
        assert_eq!(resume_identity(100, 4)["next_batch"], 0);
    }
}
