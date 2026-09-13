//! Independent, bounded trajectory controls. Every dispatch is one GPU process.
use super::*;
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Trajectory {
    schema_version: String,
    teacher: PathBuf,
    teacher_sha256: String,
    binary_sha256: String,
    output: PathBuf,
    arm: String,
    repetition: usize,
    planned_resume: bool,
    instrumented: bool,
    checkpoints: Vec<usize>,
    wall_seconds: u64,
    sealed_test: bool,
    provenance: Value,
}
fn contract(s: &Trajectory) -> Result<(), String> {
    if s.sealed_test {
        return Err("SEALED_POLICY_STOP: trajectory input".into());
    }
    if s.schema_version != "issue16-trajectory-v1"
        || !["D-A", "D-A-prime"].contains(&s.arm.as_str())
        || s.repetition > 1
        || s.checkpoints != [10, 100, 200, 400, 700, 1000]
        || s.wall_seconds != 1800
        || (s.planned_resume && s.instrumented)
        || ((s.planned_resume || s.instrumented) && s.repetition != 0)
        || !issue16::valid_sha256(&s.teacher_sha256)
    {
        return Err("CONFIGURATION_STOP: trajectory contract differs".into());
    }
    Ok(())
}
fn save(
    ctx: &Context,
    r: &SfnnTrainStepRunner,
    s: &Trajectory,
    b: &Batch,
    n: usize,
    registration: &str,
    initial: &Value,
    consumed: &[Value],
    shape: SfnnForwardShape,
    args: &crate::Args,
) -> Result<Value, String> {
    let dir = s.output.join(n.to_string());
    let tmp = s.output.join(format!(".{n}-{}.partial", std::process::id()));
    fs::create_dir(&tmp).map_err(|e| format!("RESOURCE_STOP: {e}"))?;
    let w = r.read_weights(ctx).map_err(|e| e.to_string())?;
    let o = r.read_optimizer_states(ctx).map_err(|e| e.to_string())?;
    crate::write_cuda_cpp_sfnn_weights_bin(
        &tmp.join("state.bin"),
        &w,
        PostPairwiseTransform::Identity,
        &o,
        None,
        n,
        n,
    )?;
    fs::File::open(tmp.join("state.bin")).and_then(|f| f.sync_all()).map_err(|e| format!("RESOURCE_STOP: {e}"))?;
    let device = SfnnForwardDeviceBatch::from_host(
        ctx,
        SfnnForwardHostBatch {
            stm_indices: &b.stm,
            nstm_indices: &b.nstm,
            buckets: &b.buckets,
            batch_size: b.size(),
            max_active: b.max_active,
        },
    )
    .map_err(|e| e.to_string())?;
    let workspace =
        SfnnForwardWorkspace::new(ctx, SfnnForwardWorkspaceLayout::new(shape, b.size())).map_err(|e| e.to_string())?;
    r.forward_current_weights(ctx, &device, &workspace).map_err(|e| e.to_string())?;
    let f = workspace.download(ctx).map_err(|e| e.to_string())?;
    let quantized = quantized_observation(args, &w, shape, b, &tmp.join("nn.bin"), &f.output)?;
    let row = json!({"registration_sha256":registration,"arm":s.arm,"repetition":s.repetition,"planned_resume":s.planned_resume,"instrumented":s.instrumented,"step":n,"resume":resume_identity(n,4),"initial":initial,"consumed_batches":consumed,"state_sha256":file_hash(&tmp.join("state.bin"))?,"float_outputs":f.output,"quantized":quantized,"clean_completion":true});
    publish(&tmp.join("metadata.json"), &row)?;
    if dir.exists() {
        return Err("DATA_INTEGRITY_STOP: checkpoint already exists".into());
    }
    fs::rename(tmp, dir).map_err(|e| format!("RESOURCE_STOP: {e}"))?;
    Ok(row)
}
pub fn run(path: &Path) -> Result<(), String> {
    let s: Trajectory = serde_json::from_slice(&fs::read(path).map_err(|e| format!("DATA_INTEGRITY_STOP: {e}"))?)
        .map_err(|e| format!("CONFIGURATION_STOP: {e}"))?;
    contract(&s)?;
    if file_hash(&s.teacher)? != s.teacher_sha256
        || file_hash(&std::env::current_exe().map_err(|e| e.to_string())?)? != s.binary_sha256
    {
        return Err("DATA_INTEGRITY_STOP: binary/input mismatch".into());
    }
    let args = issue16::fixed_args(&s.teacher)?;
    let seq = batches(&Spec {
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
    })?;
    let registration = file_hash(path)?;
    if s.output.exists() {
        if issue16::read_json(&s.output.join("DISPATCH.json"))?["registration_sha256"] != registration
            || s.output.join("REPORT.json").exists()
        {
            return Err("DATA_INTEGRITY_STOP: completed/mismatched dispatch".into());
        }
    } else {
        fs::create_dir_all(&s.output).map_err(|e| format!("RESOURCE_STOP: {e}"))?;
        publish(&s.output.join("DISPATCH.json"), &json!({"registration_sha256":registration,"spec":s}))?;
    }
    let ctx = Context::new(0).map_err(|e| format!("RESOURCE_STOP: {e}"))?;
    let initial_weights =
        crate::build_sfnn_initial_weights_for_cuda_cpp(&args, crate::CudaCppSfnnFeatureKind::Halfka2)?;
    let shape = initial_weights.shape;
    let selector = if s.arm == "D-A" {
        SfnnL0BackwardSelector::InverseIndex
    } else {
        SfnnL0BackwardSelector::MaterializedSparse
    };
    let mut r = runner(&ctx, initial_weights.as_host(), &seq[0], selector)?;
    drop(initial_weights);
    let initial = issue16::state_identity(&ctx, &r)?;
    let start = Instant::now();
    let mut completed = 0;
    let mut rows = Vec::new();
    let mut consumed = Vec::new();
    for &n in &s.checkpoints {
        let dir = s.output.join(n.to_string());
        if !dir.exists() {
            break;
        }
        let row = issue16::read_json(&dir.join("metadata.json"))?;
        validate_resume_identity(&row["resume"], n, 4)?;
        let expected: Vec<_> = (1..=n).map(|k| json!({"step":k,"batch_sha256":digest_json(&seq[(k-1)%4])})).collect();
        if row["registration_sha256"] != registration
            || row["initial"] != initial
            || row["arm"] != s.arm
            || row["step"] != n
            || row["consumed_batches"] != json!(expected)
            || row["state_sha256"] != file_hash(&dir.join("state.bin"))?
            || row["quantized"]["nn_bin_sha256"] != file_hash(&dir.join("nn.bin"))?
            || row["clean_completion"] != true
        {
            return Err("DATA_INTEGRITY_STOP: checkpoint identity differs".into());
        }
        completed = n;
        consumed = expected;
        rows.push(row);
    }
    if completed > 0 {
        let state = crate::load_cuda_cpp_sfnn_initial_state(
            &s.output.join(completed.to_string()).join("state.bin"),
            &args,
            crate::CudaCppSfnnFeatureKind::Halfka2,
        )?;
        if state.completed_steps != completed || state.optimizer_steps != completed {
            return Err("DATA_INTEGRITY_STOP: binary step differs".into());
        }
        r.upload_state_from_host(
            &ctx,
            state.weights.as_host(),
            state.optimizer_states.as_ref().ok_or("DATA_INTEGRITY_STOP: missing optimizer")?.as_host(),
            SfnnFactorizerActive { shared: true, ..SfnnFactorizerActive::NONE },
            SfnnFactorizerAlpha::ONE,
        )
        .map_err(|e| e.to_string())?;
        let expected = json!({"weights":hash_tensors(&weights_map(&crate::sfnn_initial_weights_into_readback(state.weights)))?,"optimizer":hash_tensors(&optimizer_host_map(state.optimizer_states.as_ref().unwrap().as_host()))?});
        if issue16::state_identity(&ctx, &r)? != expected {
            return Err("IMPLEMENTATION_STOP: resumed GPU state differs".into());
        }
    }
    for n in completed + 1..=1000 {
        if start.elapsed().as_secs() >= s.wall_seconds {
            return Err("RESOURCE_STOP: trajectory wall budget".into());
        }
        let b = &seq[(n - 1) % 4];
        if s.instrumented && (n == 1 || s.checkpoints.contains(&n)) {
            let snap = r.snapshot_device(&ctx).map_err(|e| e.to_string())?;
            let before = issue16::state_identity(&ctx, &r)?;
            step(&ctx, &mut r, b, "ordinary", false, n)?;
            let gradients = r.read_backward_trace(&ctx).map_err(|e| e.to_string())?;
            let gradient_stats = tensor_stats(gradient_map(&gradients))?;
            drop(gradients);
            r.copy_state_from_device(&ctx, &snap).map_err(|e| e.to_string())?;
            if issue16::state_identity(&ctx, &r)? != before {
                return Err("IMPLEMENTATION_STOP: measurement mutated state".into());
            }
            publish(
                &s.output.join(format!("probe-{n}-{}.json", std::process::id())),
                &json!({"step":n,"state_unchanged":true,"gradients":gradient_stats,"registration_sha256":registration}),
            )?;
        }
        step(&ctx, &mut r, b, if s.instrumented { "profile" } else { "ordinary" }, true, n)?;
        consumed.push(json!({"step":n,"batch_sha256":digest_json(b)}));
        if s.checkpoints.contains(&n) {
            eprintln!(
                "issue16 trajectory arm={} rep={} resume={} instrumented={} step={n}",
                s.arm, s.repetition, s.planned_resume, s.instrumented
            );
            rows.push(save(&ctx, &r, &s, b, n, &registration, &initial, &consumed, shape, &args)?);
            if s.planned_resume && n == 10 {
                publish(
                    &s.output.join("PLANNED_STOP.json"),
                    &json!({"typed_stop":"RESOURCE_STOP","reason":"planned process boundary at step10","registration_sha256":registration}),
                )?;
                return Err("RESOURCE_STOP: planned process boundary at step10; same command resumes".into());
            }
        }
    }
    publish(
        &s.output.join("REPORT.json"),
        &json!({"schema_version":"issue16-trajectory-report-v1","registration_sha256":registration,"binary_sha256":s.binary_sha256,"build_info":crate::bulletou_build_info_text(),"initial":initial,"rows":rows,"completed":true,"resumed_from":completed,"wall_seconds_this_attempt":start.elapsed().as_secs_f64(),"qualification_passed":false}),
    )?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_budget_and_no_factor_confound() {
        let mut s:Trajectory=serde_json::from_value(json!({"schema_version":"issue16-trajectory-v1","teacher":"x","teacher_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","binary_sha256":"x","output":"x","arm":"D-A","repetition":0,"planned_resume":false,"instrumented":false,"checkpoints":[10,100,200,400,700,1000],"wall_seconds":1800,"sealed_test":false,"provenance":{}})).unwrap();
        assert!(contract(&s).is_ok());
        s.instrumented = true;
        s.planned_resume = true;
        assert!(contract(&s).is_err());
    }
}
