//! Bounded, production-shape D-A/D-A' qualification. No ordinary training dispatch.
use bulletou_cuda_cpp::*;
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::result::Result;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

const FIXTURES: [&str; 12] = [
    "production_halfka2",
    "empty_padding",
    "high_collision",
    "stm_nstm_active_count_mismatch",
    "float_boundaries",
    "integer_export_boundaries",
    "microbatch",
    "soft_progress",
    "microbatch_soft_progress",
    "pipeline",
    "profile",
    "restore",
];
const ATOL: f64 = 1e-6;
const RTOL: f64 = 1e-5;
const SEED: u64 = 20260913;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Spec {
    schema_version: String,
    stage: String,
    teacher: PathBuf,
    teacher_sha256: String,
    binary_sha256: String,
    output: PathBuf,
    seed: u64,
    repetitions: usize,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    fixtures: Vec<String>,
    sealed_test: bool,
    provenance: Value,
    #[serde(default)]
    gate0_report: Option<PathBuf>,
    #[serde(default)]
    gate0_report_sha256: Option<String>,
}

fn file_hash(path: &Path) -> Result<String, String> {
    let mut f = fs::File::open(path).map_err(|e| format!("DATA_INTEGRITY_STOP: {}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = f.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buffer[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn digest_json(v: &impl Serialize) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(v).unwrap()))
}
fn publish(path: &Path, report: &impl Serialize) -> Result<(), String> {
    let parent = path.parent().ok_or("CONFIGURATION_STOP: output has no parent")?;
    fs::create_dir_all(parent).map_err(|e| format!("RESOURCE_STOP: {e}"))?;
    let temp = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut f =
        fs::OpenOptions::new().write(true).create_new(true).open(&temp).map_err(|e| format!("RESOURCE_STOP: {e}"))?;
    let result = (|| {
        serde_json::to_writer_pretty(&mut f, report).map_err(|e| e.to_string())?;
        f.write_all(b"\n").map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
        drop(f);
        fs::hard_link(&temp, path).map_err(|e| format!("RESOURCE_STOP: no-clobber publish {}: {e}", path.display()))
    })();
    let _ = fs::remove_file(temp);
    result
}

#[derive(Clone, Serialize)]
struct Batch {
    stm: Vec<i32>,
    nstm: Vec<i32>,
    buckets: Vec<i32>,
    targets: Vec<f32>,
    weights: Vec<f32>,
    max_active: usize,
}
impl Batch {
    fn size(&self) -> usize {
        self.buckets.len()
    }
    fn host(&self) -> SfnnTrainStepHostBatch<'_> {
        SfnnTrainStepHostBatch {
            stm_indices: &self.stm,
            nstm_indices: &self.nstm,
            buckets: &self.buckets,
            targets: &self.targets,
            entry_weights: &self.weights,
            batch_size: self.size(),
            max_active: self.max_active,
        }
    }
    fn validate(&self) -> Result<(), String> {
        if self.size() == 0
            || self.max_active == 0
            || self.stm.len() != self.size() * self.max_active
            || self.nstm.len() != self.stm.len()
            || self.targets.len() != self.size()
            || self.weights.len() != self.size()
        {
            return Err("DATA_INTEGRITY_STOP: batch dimensions".into());
        }
        for indices in [&self.stm, &self.nstm] {
            for row in indices.chunks_exact(self.max_active) {
                let mut seen = BTreeSet::new();
                let mut padding = false;
                for &feature in row {
                    if feature == -1 {
                        padding = true;
                        continue;
                    }
                    if padding || !(0..131949).contains(&feature) || !seen.insert(feature) {
                        return Err("DATA_INTEGRITY_STOP: duplicate/out-of-range/non-tail-padding feature".into());
                    }
                }
            }
        }
        if self.buckets.iter().any(|v| !(0..9).contains(v))
            || self.targets.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || self.weights.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err("DATA_INTEGRITY_STOP: invalid target/weight/bucket".into());
        }
        Ok(())
    }
}
fn batches(spec: &Spec) -> Result<Vec<Batch>, String> {
    let positions = super::read_teacher_positions_prefix(spec.teacher.to_str().ok_or("invalid teacher path")?, 64)
        .map_err(|e| e.to_string())?;
    if positions.len() != 64 {
        return Err("DATA_INTEGRITY_STOP: need exactly 64 admitted positions".into());
    }
    positions
        .chunks_exact(16)
        .map(|positions| {
            let fast = super::build_sfnn_validation_fast_batch(
                super::CudaCppSfnnFeatureKind::Halfka2,
                super::LayerStackMode::Kingrank3by3,
                positions,
                None,
            )?;
            let batch = Batch {
                stm: fast.stm,
                nstm: fast.nstm,
                buckets: fast.buckets,
                targets: positions.iter().map(|p| 1.0 / (1.0 + (-(p.score() as f32) / 600.0).exp())).collect(),
                weights: vec![1.0; 16],
                max_active: fast.layout.max_active,
            };
            batch.validate()?;
            Ok(batch)
        })
        .collect()
}

fn weights_map(w: &SfnnTrainWeightsReadback) -> BTreeMap<String, &[f32]> {
    let mut out = BTreeMap::new();
    macro_rules! base { ($($n:ident),*) => { $(out.insert(stringify!($n).into(), w.$n.as_slice());)* }; }
    base!(l0w, l0b, l1w, l1b, l2w, l2b, l3w, l3b);
    macro_rules! opt { ($($n:ident),*) => { $(if let Some(v)=&w.$n { out.insert(stringify!($n).into(),v.as_slice()); })* }; }
    opt!(l1fw, l1fb, l2fw, l2fb, l3fw, l3fb);
    out
}
fn gradient_map(w: &SfnnBackwardReadback) -> BTreeMap<String, &[f32]> {
    let mut out = BTreeMap::new();
    macro_rules! base { ($($n:ident),*) => { $(out.insert(stringify!($n).trim_end_matches("_gradients").into(), w.$n.as_slice());)* }; }
    base!(
        l0w_gradients,
        l0b_gradients,
        l1w_gradients,
        l1b_gradients,
        l1fw_gradients,
        l1fb_gradients,
        l2w_gradients,
        l2b_gradients,
        l2fw_gradients,
        l2fb_gradients,
        l3w_gradients,
        l3b_gradients,
        l3fw_gradients,
        l3fb_gradients
    );
    out
}
fn optimizer_map(w: &SfnnRangerOptimizerStatesReadback) -> BTreeMap<String, &[f32]> {
    let mut out = BTreeMap::new();
    macro_rules! state {
        ($name:expr, $v:expr) => {{
            let v = $v;
            out.insert(format!("{}.momentum", $name), v.momentum.as_slice());
            out.insert(format!("{}.velocity", $name), v.velocity.as_slice());
            out.insert(format!("{}.slow", $name), v.slow_params.as_slice());
        }};
    }
    macro_rules! base { ($($n:ident),*) => { $(state!(stringify!($n), &w.$n);)* }; }
    base!(l0w, l0b, l1w, l1b, l2w, l2b, l3w, l3b);
    macro_rules! opt { ($($n:ident),*) => { $(if let Some(v)=&w.$n { state!(stringify!($n),v); })* }; }
    opt!(l1fw, l1fb, l2fw, l2fb, l3fw, l3fb);
    out
}
fn shape(name: &str, len: usize) -> Vec<usize> {
    match name.split('.').next().unwrap() {
        "stm_l0" | "nstm_l0" | "pairwise_l1_input" => vec![16, 1024],
        "l1" => vec![16, 8],
        "l2_input" => vec![16, 14],
        "l2" => vec![16, 64],
        "l0w" => vec![133578, 1024],
        "l0b" => vec![1024],
        "l1w" => vec![9, 8, 1024],
        "l1b" => vec![9, 8],
        "l1fw" => vec![8, 1024],
        "l1fb" => vec![8],
        "l2w" => vec![9, 64, 14],
        "l2b" => vec![9, 64],
        "l2fw" => vec![64, 14],
        "l2fb" => vec![64],
        "l3w" => vec![9, 64],
        "l3b" => vec![9],
        "l3fw" => vec![64],
        "l3fb" => vec![1],
        _ => vec![len],
    }
}
fn hash_tensors(tensors: &BTreeMap<String, &[f32]>) -> Result<String, String> {
    let mut h = Sha256::new();
    for (name, values) in tensors {
        h.update(name.as_bytes());
        h.update((values.len() as u64).to_le_bytes());
        // Host is little endian on the qualified platform; explicit bytes keep the digest portable.
        for chunk in values.chunks(16384) {
            let mut bytes = Vec::with_capacity(chunk.len() * 4);
            for v in chunk {
                if !v.is_finite() {
                    return Err("IMPLEMENTATION_STOP: non-finite state".into());
                }
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            h.update(bytes);
        }
    }
    Ok(format!("{:x}", h.finalize()))
}
fn metric(a: &[f32], b: &[f32], dims: Vec<usize>) -> Result<Value, String> {
    metric_tolerance(a, b, dims, ATOL, RTOL)
}
fn metric_tolerance(a: &[f32], b: &[f32], dims: Vec<usize>, atol: f64, rtol: f64) -> Result<Value, String> {
    if a.len() != b.len() || a.is_empty() || dims.iter().product::<usize>() != a.len() {
        return Err("IMPLEMENTATION_STOP: tensor shape mismatch".into());
    }
    let (mut abs, mut rel, mut sq, mut aa, mut bb, mut dot, mut violations) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0u64);
    let mut max_scaled_error = 0.0f64;
    for (&a, &b) in a.iter().zip(b) {
        if !a.is_finite() || !b.is_finite() {
            return Err("IMPLEMENTATION_STOP: non-finite tensor".into());
        }
        let (a, b) = (a as f64, b as f64);
        let d = (a - b).abs();
        let scale = a.abs().max(b.abs());
        abs = abs.max(d);
        rel = rel.max(d / scale.max(1e-30));
        sq += d * d;
        aa += a * a;
        bb += b * b;
        dot += a * b;
        max_scaled_error = max_scaled_error.max(d / (atol + rtol * scale));
        violations += u64::from(d > atol + rtol * scale);
    }
    Ok(
        json!({"shape":dims,"count":a.len(),"max_abs":abs,"max_rel":rel,"rms":(sq/a.len() as f64).sqrt(),"relative_l2":sq.sqrt()/aa.sqrt().max(bb.sqrt()).max(1e-30),"cosine":if aa==0.0 && bb==0.0 {1.0} else {(dot/(aa*bb).sqrt().max(1e-30)).clamp(-1.0,1.0)},"absolute_tolerance":atol,"relative_tolerance":rtol,"max_scaled_error":max_scaled_error,"violations":violations,"passed":violations==0}),
    )
}
fn compare(a: BTreeMap<String, &[f32]>, b: BTreeMap<String, &[f32]>) -> Result<Value, String> {
    compare_tolerance(a, b, ATOL, RTOL)
}
fn compare_tolerance(
    a: BTreeMap<String, &[f32]>,
    b: BTreeMap<String, &[f32]>,
    atol: f64,
    rtol: f64,
) -> Result<Value, String> {
    if a.keys().ne(b.keys()) {
        return Err("IMPLEMENTATION_STOP: tensor set mismatch".into());
    }
    let mut out = serde_json::Map::new();
    for (name, v) in a {
        out.insert(name.clone(), metric_tolerance(v, b[&name], shape(&name, v.len()), atol, rtol)?);
    }
    Ok(Value::Object(out))
}
fn all_passed(v: &Value) -> bool {
    v.as_object().is_some_and(|o| {
        !o.is_empty()
            && o.values().all(|m| {
                let numeric = ["max_abs", "max_rel", "rms", "relative_l2", "max_scaled_error"];
                let valid = numeric.iter().all(|k| m[*k].as_f64().is_some_and(|v| v.is_finite() && v >= 0.0));
                valid
                    && m["cosine"].as_f64().is_some_and(|v| v.is_finite() && (-1.0..=1.0).contains(&v))
                    && m["count"].as_u64().is_some_and(|n| n > 0)
                    && m["max_scaled_error"].as_f64().is_some_and(|v| v <= 1.0)
                    && m["violations"].as_u64() == Some(0)
                    && m["passed"] == true
            })
    })
}
fn forward_map<'a>(f: &'a SfnnForwardReadback, l: &'a ScalarLossReadback) -> BTreeMap<String, &'a [f32]> {
    [
        ("stm_l0", f.stm_l0.as_slice()),
        ("nstm_l0", &f.nstm_l0),
        ("pairwise_l1_input", &f.combined),
        ("l1", &f.l1),
        ("l2_input", &f.l2_input),
        ("l2", &f.l2),
        ("output", &f.output),
        ("loss_mean", std::slice::from_ref(&l.mean)),
        ("loss_weighted_sum", std::slice::from_ref(&l.weighted_sum)),
        ("loss_per_sample", &l.per_sample),
        ("output_gradient", &l.mean_output_gradients),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), v))
    .collect()
}
fn parameters(step: usize) -> RangerUpdateParams {
    RangerUpdateParams {
        radam: RAdamUpdateParams {
            step: step as u64,
            learning_rate: 0.000875,
            min_weight: -1.98,
            max_weight: 1.98,
            ..Default::default()
        },
        lookahead_alpha: 0.5,
        lookahead_period: 6,
    }
}
fn runner(
    ctx: &Context,
    w: SfnnForwardHostWeights<'_>,
    b: &Batch,
    selector: SfnnL0BackwardSelector,
) -> Result<SfnnTrainStepRunner, String> {
    SfnnTrainStepRunner::new_with_factorizer_and_l0_backward_selector(
        ctx,
        w,
        b.size(),
        b.max_active,
        SfnnFactorizerActive { shared: true, ..SfnnFactorizerActive::NONE },
        SfnnFactorizerAlpha::ONE,
        selector,
    )
    .map_err(|e| e.to_string())
}
fn step(
    ctx: &Context,
    r: &mut SfnnTrainStepRunner,
    b: &Batch,
    kind: &str,
    update: bool,
    n: usize,
) -> Result<(), String> {
    let loss = ScalarLossKind::SigmoidPow { pow_exp: 2.0 };
    if kind.contains("soft_progress") {
        let b_b: Vec<i32> = b.buckets.iter().map(|v| (v + 1) % 9).collect();
        let interp = vec![0.375; b.size()];
        r.step_soft_progress_no_readback_with_update_lr_multipliers_and_dirty_buckets(
            ctx,
            parameters(n),
            loss,
            1.0,
            SfnnSoftProgressTrainStepHostBatch {
                stm_indices: &b.stm,
                nstm_indices: &b.nstm,
                buckets_a: &b.buckets,
                buckets_b: &b_b,
                interpolation: &interp,
                targets: &b.targets,
                entry_weights: &b.weights,
                batch_size: b.size(),
                max_active: b.max_active,
            },
            update,
            SfnnLayerLrMultipliers::default(),
            None,
        )
        .map_err(|e| e.to_string())?;
    } else if kind == "pipeline" {
        let upload = Context::new(0).map_err(|e| e.to_string())?;
        r.step_pipelined_no_readback_with_loss_finalize_and_update(
            ctx,
            &upload,
            parameters(n),
            loss,
            1.0,
            b.host(),
            true,
            update,
        )
        .map_err(|e| e.to_string())?;
    } else if kind == "profile" {
        r.step_profiled_no_readback_with_update(ctx, parameters(n), loss, 1.0, b.host(), update)
            .map_err(|e| e.to_string())?;
    } else {
        r.step_no_readback_with_loss_finalize_and_update(ctx, parameters(n), loss, 1.0, b.host(), true, update)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}
fn execute(ctx: &Context, r: &mut SfnnTrainStepRunner, seq: &[Batch], kind: &str, update: bool) -> Result<(), String> {
    if kind == "restore" {
        let snapshot = r.snapshot_device(ctx).map_err(|e| e.to_string())?;
        step(ctx, r, &seq[1], "ordinary", true, 1)?;
        r.copy_state_from_device(ctx, &snapshot).map_err(|e| e.to_string())?;
    }
    if kind.contains("microbatch") {
        step(ctx, r, &seq[0], kind, false, 1)?;
        step(ctx, r, &seq[1], kind, update, 1)
    } else {
        step(ctx, r, &seq[0], kind, update, 1)
    }
}

fn fixture_batches(base: &[Batch], name: &str) -> Vec<Batch> {
    let mut seq = base[..2].to_vec();
    let b = &mut seq[0];
    match name {
        "empty_padding" | "float_boundaries" | "integer_export_boundaries" => {
            b.stm.fill(-1);
            b.nstm.fill(-1);
        }
        "high_collision" => {
            let stm = b.stm[..b.max_active].to_vec();
            let nstm = b.nstm[..b.max_active].to_vec();
            for row in b.stm.chunks_exact_mut(b.max_active) {
                row.copy_from_slice(&stm);
            }
            for row in b.nstm.chunks_exact_mut(b.max_active) {
                row.copy_from_slice(&nstm);
            }
        }
        "stm_nstm_active_count_mismatch" => {
            for row in b.nstm.chunks_exact_mut(b.max_active) {
                row[1..].fill(-1);
            }
        }
        _ => {}
    }
    seq
}
fn boundary_weights(weights: &mut super::CudaCppSfnnInitialWeights, name: &str) {
    if name == "float_boundaries" || name == "integer_export_boundaries" {
        weights.l0w.fill(0.0);
        let eps = 4.0 * f32::EPSILON;
        let bounds = [-eps, 0.0, eps, 63.0 / 128.0 - eps, 63.0 / 128.0, 63.0 / 128.0 + eps, 1.0 - eps, 1.0, 1.0 + eps];
        for (i, v) in weights.l0b.iter_mut().enumerate() {
            *v = if name == "float_boundaries" {
                if i < 512 { bounds[i % bounds.len()] } else { 0.75 }
            } else {
                [0.0, 63.0 / 127.0, 1.0][i % 3]
            };
        }
    }
}
fn export_boundaries(
    args: &super::Args,
    ctx: &Context,
    w: &super::CudaCppSfnnInitialWeights,
    b: &Batch,
    dir: &Path,
) -> Result<Value, String> {
    let r = runner(ctx, w.as_host(), b, SfnnL0BackwardSelector::Auto)?;
    let state = r.read_weights(ctx).map_err(|e| e.to_string())?;
    drop(r);
    let path = dir.join("boundary.nn.bin");
    super::write_cuda_cpp_sfnn_nn_bin(
        &path,
        super::CudaCppSfnnFeatureKind::Halfka2,
        w.shape,
        &state,
        super::effective_sfnn_factorizer_spec(args),
        super::SfnnFactorizerAlphaSpec::ONE,
        None,
        None,
        None,
    )?;
    let parsed = super::parse_quantized_sfnn_nn_bin(&path, args.arch(), super::LayerStackMode::Kingrank3by3)?;
    let expected: Vec<i16> = (0..1024).map(|i| [0, 63, 127][i % 3]).collect();
    let fast = bulletou_lib::value::FastBatchHost {
        layout: bulletou_lib::value::FastBatchLayout {
            batch_size: b.size(),
            max_active: b.max_active,
            output_size: 1,
            hand_count_dim: 0,
        },
        stm: b.stm.clone(),
        nstm: b.nstm.clone(),
        buckets: b.buckets.clone(),
        targets: b.targets.clone(),
        weights: b.weights.clone(),
        hand_count: None,
        progress: None,
    };
    let mut trace = super::QuantizedSfnnThreadState::new(&parsed);
    let output = super::quantized_sfnn_forward_sample(
        &parsed,
        &fast,
        0,
        7,
        super::QuantizedRoundMode::Floor,
        super::QuantizedRoundMode::Floor,
        super::QuantizedRoundMode::Floor,
        &mut trace,
    )?;
    let expected_ft: Vec<u8> = (0..1024)
        .map(|i| {
            let p = i % 512;
            ((i32::from(expected[p]) * i32::from(expected[512 + p])) / 128) as u8
        })
        .collect();
    let pass = parsed.l0b == expected && parsed.l0w.iter().all(|&v| v == 0) && trace.ft == expected_ft;
    let result = json!({"passed":pass,"nn_bin_sha256":file_hash(&path)?,"l0b_values":&parsed.l0b[..9],"integer_ft_values":&trace.ft[..9],"raw_integer_output":output.raw,"producer":"write_cuda_cpp_sfnn_nn_bin -> parse_quantized_sfnn_nn_bin"});
    fs::remove_file(path).map_err(|e| e.to_string())?;
    Ok(result)
}

fn l0_reference(f: &SfnnForwardReadback, g: &SfnnBackwardReadback, b: &Batch) -> Result<Value, String> {
    let mut rows: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    let mut bias = vec![0.0f64; 1024];
    for sample in 0..b.size() {
        for (perspective, (activation, indices)) in [(&f.stm_l0, &b.stm), (&f.nstm_l0, &b.nstm)].into_iter().enumerate()
        {
            for row in 0..1024 {
                let value = activation[sample * 1024 + row];
                if value <= 0.0 || value >= 1.0 {
                    continue;
                }
                let mate = (row + 512) % 1024;
                let pair = row % 512;
                let grad = g.combined_gradients[sample * 1024 + perspective * 512 + pair] as f64
                    * activation[sample * 1024 + mate] as f64
                    * (127.0 / 128.0);
                bias[row] += grad;
                for &feature in &indices[sample * b.max_active..(sample + 1) * b.max_active] {
                    if feature >= 0 {
                        for index in [feature as usize, 131949 + feature as usize % 1629] {
                            rows.entry(index).or_insert_with(|| vec![0.0; 1024])[row] += grad;
                        }
                    }
                }
            }
        }
    }
    let mut expected = vec![0.0f32; 133578 * 1024];
    for (feature, values) in rows {
        for (row, value) in values.into_iter().enumerate() {
            expected[feature * 1024 + row] = value as f32;
        }
    }
    let bias: Vec<f32> = bias.into_iter().map(|v| v as f32).collect();
    Ok(
        json!({"l0w":metric(&g.l0w_gradients,&expected,vec![133578,1024])?,"l0b":metric(&g.l0b_gradients,&bias,vec![1024])?}),
    )
}

fn gate0(args: &super::Args, spec: &Spec) -> Result<Value, String> {
    let seq = batches(spec)?;
    let ctx = Context::new(args.cuda_cpp_device).map_err(|e| format!("RESOURCE_STOP: {e}"))?;
    let mut results = Vec::new();
    for repetition in 0..spec.repetitions {
        for name in FIXTURES {
            let existing = spec.output.join(format!("{repetition}-{name}.json"));
            if existing.exists() {
                let row: Value = serde_json::from_slice(&fs::read(&existing).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
                if row["spec_sha256"] != digest_json(spec)
                    || row["fixture"] != name
                    || row["repetition"] != repetition
                    || row["fixture_sha256"] != digest_json(&fixture_batches(&seq, name))
                {
                    return Err("DATA_INTEGRITY_STOP: resumed fixture identity differs".into());
                }
                results.push(row);
                continue;
            }
            let start = Instant::now();
            eprintln!("issue13 gate0 repetition={repetition} fixture={name}");
            let batch_seq = fixture_batches(&seq, name);
            for b in &batch_seq {
                b.validate()?;
            }
            let mut weights =
                super::build_sfnn_initial_weights_for_cuda_cpp(args, super::CudaCppSfnnFeatureKind::Halfka2)?;
            boundary_weights(&mut weights, name);
            let mut a = runner(&ctx, weights.as_host(), &batch_seq[0], SfnnL0BackwardSelector::Auto)?;
            let initial_weights = a.read_weights(&ctx).map_err(|e| e.to_string())?;
            let initial_opt = a.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
            let initial_identity = json!({"weights_sha256":hash_tensors(&weights_map(&initial_weights))?,"optimizer_sha256":hash_tensors(&optimizer_map(&initial_opt))?,"scheduler":{"step":0,"lr":0.000875},"rng":{"seed":SEED,"runtime_draws":0,"initialization":"production fixed per-tensor seeds"},"dataloader":{"next_batch":0,"sequence_sha256":digest_json(&batch_seq)},"pending_gradients":false});
            drop(initial_weights);
            drop(initial_opt);
            execute(&ctx, &mut a, &batch_seq, name, false)?;
            let af = a.read_forward_trace(&ctx).map_err(|e| e.to_string())?;
            let al = a.read_loss(&ctx).map_err(|e| e.to_string())?;
            let ag = a.read_backward_trace(&ctx).map_err(|e| e.to_string())?;
            let a_path = format!("{:?}", a.l0_backward_path());
            drop(a);
            let mut b = runner(&ctx, weights.as_host(), &batch_seq[0], SfnnL0BackwardSelector::MaterializedSparse)?;
            let b_initial_w = b.read_weights(&ctx).map_err(|e| e.to_string())?;
            let b_initial_o = b.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
            if json!(hash_tensors(&weights_map(&b_initial_w))?) != initial_identity["weights_sha256"]
                || json!(hash_tensors(&optimizer_map(&b_initial_o))?) != initial_identity["optimizer_sha256"]
            {
                return Err("IMPLEMENTATION_STOP: initial D-A/D-A-prime state mismatch".into());
            }
            drop(b_initial_w);
            drop(b_initial_o);
            b.fill_l0_gradient_sentinel(&ctx, 17.0, -23.0).map_err(|e| e.to_string())?;
            let before = b.read_backward_trace(&ctx).map_err(|e| e.to_string())?;
            let sentinel_before =
                before.l0w_gradients.iter().all(|&v| v == 17.0) && before.l0b_gradients.iter().all(|&v| v == -23.0);
            drop(before);
            execute(&ctx, &mut b, &batch_seq, name, false)?;
            let bf = b.read_forward_trace(&ctx).map_err(|e| e.to_string())?;
            let bl = b.read_loss(&ctx).map_err(|e| e.to_string())?;
            let bg = b.read_backward_trace(&ctx).map_err(|e| e.to_string())?;
            let b_path = format!("{:?}", b.l0_backward_path());
            drop(b);
            let forward = compare(forward_map(&af, &al), forward_map(&bf, &bl))?;
            let cpu_reference = if !name.contains("microbatch") && !name.contains("soft_progress") {
                json!({"D-A":l0_reference(&af,&ag,&batch_seq[0])?,"D-A-prime":l0_reference(&bf,&bg,&batch_seq[0])?})
            } else {
                Value::Null
            };
            let cpu_pass = cpu_reference.is_null()
                || (all_passed(&cpu_reference["D-A"]) && all_passed(&cpu_reference["D-A-prime"]));
            let gradients = compare(gradient_map(&ag), gradient_map(&bg))?;
            let sentinel = sentinel_before && gradients["l0w"]["passed"] == true && gradients["l0b"]["passed"] == true;
            let float_bounds = if name == "float_boundaries" {
                let expected: Vec<f32> = weights.l0b.iter().map(|v| v.clamp(0.0, 1.0)).collect();
                metric(&af.stm_l0[..1024], &expected, vec![1024])?
            } else {
                Value::Null
            };
            drop(ag);
            drop(bg);
            drop(af);
            drop(bf);
            drop(al);
            drop(bl);
            let mut a = runner(&ctx, weights.as_host(), &batch_seq[0], SfnnL0BackwardSelector::Auto)?;
            execute(&ctx, &mut a, &batch_seq, name, true)?;
            let aw = a.read_weights(&ctx).map_err(|e| e.to_string())?;
            let ao = a.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
            drop(a);
            let mut b = runner(&ctx, weights.as_host(), &batch_seq[0], SfnnL0BackwardSelector::MaterializedSparse)?;
            execute(&ctx, &mut b, &batch_seq, name, true)?;
            let bw = b.read_weights(&ctx).map_err(|e| e.to_string())?;
            let bo = b.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
            drop(b);
            let parameters = compare(weights_map(&aw), weights_map(&bw))?;
            let optimizer = compare(optimizer_map(&ao), optimizer_map(&bo))?;
            drop(aw);
            drop(ao);
            drop(bw);
            drop(bo);
            let integer_bounds = if name == "integer_export_boundaries" {
                export_boundaries(args, &ctx, &weights, &batch_seq[0], &spec.output)?
            } else {
                Value::Null
            };
            let passed = cpu_pass
                && all_passed(&forward)
                && all_passed(&gradients)
                && all_passed(&parameters)
                && all_passed(&optimizer)
                && sentinel
                && (float_bounds.is_null() || float_bounds["passed"] == true)
                && (integer_bounds.is_null() || integer_bounds["passed"] == true);
            let row = json!({"spec_sha256":digest_json(spec),"fixture":name,"repetition":repetition,"fixture_sha256":digest_json(&batch_seq),"initial_state":initial_identity,"initial_state_sha256":digest_json(&initial_identity),"kernel_paths":{"D-A":a_path,"D-A-prime":b_path},"sentinel":{"before_verified":sentinel_before,"replacement_verified":sentinel},"cpu_f64_l0_reference":cpu_reference,"forward":forward,"gradients":gradients,"parameters":parameters,"optimizer":optimizer,"float_boundaries":float_bounds,"integer_export_boundaries":integer_bounds,"elapsed_seconds":start.elapsed().as_secs_f64(),"passed":passed});
            publish(&spec.output.join(format!("{repetition}-{name}.json")), &row)?;
            results.push(row);
        }
    }
    let passed = results.iter().all(|v| v["passed"] == true);
    Ok(
        json!({"schema_version":"issue13-gate0-v2","passed":passed,"typed_stop":if passed {Value::Null}else{json!("IMPLEMENTATION_STOP")},"fixtures":results}),
    )
}

fn stats(values: &[f32]) -> Result<Value, String> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return Err("IMPLEMENTATION_STOP: invalid statistics input".into());
    }
    let n = values.len() as f64;
    let sum: f64 = values.iter().map(|v| *v as f64).sum();
    let abs: f64 = values.iter().map(|v| v.abs() as f64).sum();
    let sq: f64 = values.iter().map(|v| (*v as f64).powi(2)).sum();
    Ok(
        json!({"count":values.len(),"mean":sum/n,"mean_absolute":abs/n,"rms":(sq/n).sqrt(),"zero_fraction":values.iter().filter(|v|**v==0.0).count() as f64/n,"saturation_fraction":values.iter().filter(|v|**v>=127.0/128.0).count() as f64/n}),
    )
}
fn tensor_stats(values: BTreeMap<String, &[f32]>) -> Result<Value, String> {
    let mut out = serde_json::Map::new();
    for (name, v) in values {
        out.insert(name, stats(v)?);
    }
    Ok(Value::Object(out))
}
fn optimizer_host_map<'a>(w: SfnnRangerOptimizerHostStates<'a>) -> BTreeMap<String, &'a [f32]> {
    let mut out = BTreeMap::new();
    macro_rules! state {
        ($name:expr,$v:expr) => {{
            let v = $v;
            out.insert(format!("{}.momentum", $name), v.momentum);
            out.insert(format!("{}.velocity", $name), v.velocity);
            out.insert(format!("{}.slow", $name), v.slow_params);
        }};
    }
    macro_rules! base {($($n:ident),*)=>{$(state!(stringify!($n),w.$n);)*};}
    base!(l0w, l0b, l1w, l1b, l2w, l2b, l3w, l3b);
    macro_rules! opt {($($n:ident),*)=>{$(if let Some(v)=w.$n{state!(stringify!($n),v);})*};}
    opt!(l1fw, l1fb, l2fw, l2fb, l3fw, l3fb);
    out
}
fn fast_batch(b: &Batch) -> bulletou_lib::value::FastBatchHost {
    bulletou_lib::value::FastBatchHost {
        layout: bulletou_lib::value::FastBatchLayout {
            batch_size: b.size(),
            max_active: b.max_active,
            output_size: 1,
            hand_count_dim: 0,
        },
        stm: b.stm.clone(),
        nstm: b.nstm.clone(),
        buckets: b.buckets.clone(),
        targets: b.targets.clone(),
        weights: b.weights.clone(),
        hand_count: None,
        progress: None,
    }
}
fn quantized_observation(
    args: &super::Args,
    w: &SfnnTrainWeightsReadback,
    shape: SfnnForwardShape,
    b: &Batch,
    path: &Path,
    float: &[f32],
) -> Result<Value, String> {
    super::write_cuda_cpp_sfnn_nn_bin(
        path,
        super::CudaCppSfnnFeatureKind::Halfka2,
        shape,
        w,
        super::effective_sfnn_factorizer_spec(args),
        super::SfnnFactorizerAlphaSpec::ONE,
        None,
        None,
        None,
    )?;
    let parsed = super::parse_quantized_sfnn_nn_bin(path, args.arch(), super::LayerStackMode::Kingrank3by3)?;
    let fast = fast_batch(b);
    let mut state = super::QuantizedSfnnThreadState::new(&parsed);
    let mut outputs = Vec::new();
    for i in 0..b.size() {
        let v = super::quantized_sfnn_forward_sample(
            &parsed,
            &fast,
            i,
            7,
            super::QuantizedRoundMode::Floor,
            super::QuantizedRoundMode::Floor,
            super::QuantizedRoundMode::Floor,
            &mut state,
        )?;
        outputs.push(v.raw as f32 / super::quantized_sfnn_raw_output_scale());
    }
    let flips = outputs.iter().zip(float).filter(|(q, f)| q.signum() != f.signum()).count();
    Ok(
        json!({"nn_bin_sha256":file_hash(path)?,"ft_weight_nonzero_fraction":parsed.l0w.iter().filter(|v|**v!=0).count() as f64/parsed.l0w.len() as f64,"ft_bias_nonzero_fraction":parsed.l0b.iter().filter(|v|**v!=0).count() as f64/parsed.l0b.len() as f64,"output":stats(&outputs)?,"outputs":outputs,"fp32_sign_flip_fraction":flips as f64/b.size() as f64}),
    )
}
fn validate_gate0_evidence(spec: &Spec) -> Result<(), String> {
    let path = spec.gate0_report.as_ref().ok_or("CONFIGURATION_STOP: Gate 1 requires Gate 0 report")?;
    if Some(file_hash(path)?) != spec.gate0_report_sha256 {
        return Err("DATA_INTEGRITY_STOP: Gate 0 digest mismatch".into());
    }
    let report: Value =
        serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if report["schema_version"] != "issue13-gate0-v2"
        || report["binary_sha256"] != spec.binary_sha256
        || report["passed"] != true
        || !report["typed_stop"].is_null()
    {
        return Err("IMPLEMENTATION_STOP: Gate 0 did not pass on this binary".into());
    }
    let dispatch: Value = serde_json::from_slice(
        &fs::read(path.parent().ok_or("missing Gate 0 parent")?.join("DISPATCH.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if dispatch["config"]["teacher_sha256"] != spec.teacher_sha256
        || dispatch["preregistration_sha256"] != report["preregistration_sha256"]
        || report["provenance"] != spec.provenance
    {
        return Err("DATA_INTEGRITY_STOP: Gate 0 registration differs".into());
    }
    let rows = report["fixtures"].as_array().ok_or("IMPLEMENTATION_STOP: no Gate 0 fixture evidence")?;
    if rows.len() != FIXTURES.len() * 2 {
        return Err("IMPLEMENTATION_STOP: incomplete Gate 0 fixture set".into());
    }
    let base = batches(spec)?;
    for repetition in 0..2 {
        for name in FIXTURES {
            let matches: Vec<_> =
                rows.iter().filter(|v| v["fixture"] == name && v["repetition"] == repetition).collect();
            if matches.len() != 1 {
                return Err("IMPLEMENTATION_STOP: missing or duplicate Gate 0 fixture".into());
            }
            let row = matches[0];
            if row["fixture_sha256"] != digest_json(&fixture_batches(&base, name))
                || row["passed"] != true
                || row["sentinel"]["replacement_verified"] != true
            {
                return Err("DATA_INTEGRITY_STOP: Gate 0 fixture identity or sentinel mismatch".into());
            }
            if row["kernel_paths"] != json!({"D-A":"InverseIndex","D-A-prime":"MaterializedSparse"}) {
                return Err("IMPLEMENTATION_STOP: wrong kernel evidence".into());
            }
            for (key, count) in [("forward", 11), ("gradients", 14), ("parameters", 14), ("optimizer", 42)] {
                if row[key].as_object().map(|m| m.len()) != Some(count) {
                    return Err("IMPLEMENTATION_STOP: tensor evidence missing".into());
                }
            }
            if !name.contains("microbatch")
                && !name.contains("soft_progress")
                && (!all_passed(&row["cpu_f64_l0_reference"]["D-A"])
                    || !all_passed(&row["cpu_f64_l0_reference"]["D-A-prime"]))
            {
                return Err("IMPLEMENTATION_STOP: CPU reference failed".into());
            }
            if name == "float_boundaries" && row["float_boundaries"]["passed"] != true
                || name == "integer_export_boundaries" && row["integer_export_boundaries"]["passed"] != true
            {
                return Err("IMPLEMENTATION_STOP: boundary evidence missing".into());
            }
            for key in ["forward", "gradients", "parameters", "optimizer"] {
                if !all_passed(&row[key]) {
                    return Err("IMPLEMENTATION_STOP: Gate 0 tensor comparison failed".into());
                }
            }
        }
    }
    Ok(())
}
fn resume_identity(step: usize, batches: usize) -> Value {
    json!({"completed_steps":step,"optimizer_steps":step,"lr":0.000875,
           "next_batch":step%batches,"rng_seed":SEED,"runtime_rng_draws":0,"pending_gradients":false})
}
fn validate_resume_identity(value: &Value, step: usize, batches: usize) -> Result<(), String> {
    if value != &resume_identity(step, batches) {
        return Err("DATA_INTEGRITY_STOP: checkpoint scheduler/RNG/cursor state differs".into());
    }
    Ok(())
}
fn gate1(args: &super::Args, spec: &Spec) -> Result<Value, String> {
    validate_gate0_evidence(spec)?;
    let seq = batches(spec)?;
    let ctx = Context::new(0).map_err(|e| format!("RESOURCE_STOP: {e}"))?;
    let checkpoints = [1usize, 10, 100, 1000];
    let mut arm_reports = serde_json::Map::new();
    let mut comparisons = serde_json::Map::new();
    let mut common_initial: Option<Value> = None;
    for (arm, selector) in
        [("D-A", SfnnL0BackwardSelector::Auto), ("D-A-prime", SfnnL0BackwardSelector::MaterializedSparse)]
    {
        let arm_dir = spec.output.join(arm);
        fs::create_dir_all(&arm_dir).map_err(|e| e.to_string())?;
        let initial = super::build_sfnn_initial_weights_for_cuda_cpp(args, super::CudaCppSfnnFeatureKind::Halfka2)?;
        let shape = initial.shape;
        let mut r = runner(&ctx, initial.as_host(), &seq[0], selector)?;
        let iw = r.read_weights(&ctx).map_err(|e| e.to_string())?;
        let io = r.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
        let identity = json!({"weights":hash_tensors(&weights_map(&iw))?,"optimizer":hash_tensors(&optimizer_map(&io))?,"scheduler":{"step":0,"lr":0.000875},"rng":{"seed":SEED,"runtime_draws":0},"dataloader":{"next_batch":0,"sequence_sha256":digest_json(&seq)},"pending_gradients":false});
        drop(iw);
        drop(io);
        if let Some(expected) = &common_initial {
            if expected != &identity {
                return Err("DATA_INTEGRITY_STOP: paired complete initial state differs".into());
            }
        } else {
            common_initial = Some(identity.clone());
        }
        let mut consumed = Vec::new();
        let mut saved = serde_json::Map::new();
        let mut total_kernel_ms = 0.0f64;
        let arm_started = Instant::now();
        let mut completed = 0usize;
        for n in checkpoints {
            let dir = arm_dir.join(n.to_string());
            if !dir.exists() {
                break;
            }
            let row: Value = serde_json::from_slice(&fs::read(dir.join("metadata.json")).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            validate_resume_identity(&row["resume"], n, seq.len())?;
            let expected_batches: Vec<Value> =
                (1..=n).map(|step| json!({"step":step,"batch_sha256":digest_json(&seq[(step-1)%seq.len()])})).collect();
            if row["spec_sha256"] != digest_json(spec)
                || row["arm"] != arm
                || row["initial_state_sha256"] != digest_json(&identity)
                || row["state_sha256"] != file_hash(&dir.join("state.bin"))?
                || row["quantized"]["nn_bin_sha256"] != file_hash(&dir.join("nn.bin"))?
                || row["consumed_batches"] != json!(expected_batches)
                || row["resume"]["completed_steps"] != n
                || row["resume"]["optimizer_steps"] != n
                || row["resume"]["pending_gradients"] != false
            {
                return Err("DATA_INTEGRITY_STOP: resume checkpoint mismatch".into());
            }
            completed = n;
            consumed = expected_batches;
            total_kernel_ms = row["timing"]["total_gpu_ms"].as_f64().ok_or("invalid checkpoint timing")?;
            saved.insert(n.to_string(), row);
        }
        if completed > 0 {
            let loaded = super::load_cuda_cpp_sfnn_initial_state(
                &arm_dir.join(completed.to_string()).join("state.bin"),
                args,
                super::CudaCppSfnnFeatureKind::Halfka2,
            )?;
            if loaded.completed_steps != completed || loaded.optimizer_steps != completed {
                return Err("DATA_INTEGRITY_STOP: state step mismatch".into());
            }
            r.upload_state_from_host(
                &ctx,
                loaded.weights.as_host(),
                loaded.optimizer_states.as_ref().ok_or("missing resume optimizer")?.as_host(),
                SfnnFactorizerActive { shared: true, ..SfnnFactorizerActive::NONE },
                SfnnFactorizerAlpha::ONE,
            )
            .map_err(|e| e.to_string())?;
        }
        for n in completed + 1..=1000 {
            let b = &seq[(n - 1) % seq.len()];
            let checkpoint = checkpoints.contains(&n);
            let mut gradient_stats = Value::Null;
            let mut before = None;
            if checkpoint {
                let snap = r.snapshot_device(&ctx).map_err(|e| e.to_string())?;
                before = Some(r.read_weights(&ctx).map_err(|e| e.to_string())?);
                step(&ctx, &mut r, b, "ordinary", false, n)?;
                let grad = r.read_backward_trace(&ctx).map_err(|e| e.to_string())?;
                gradient_stats = tensor_stats(gradient_map(&grad))?;
                drop(grad);
                r.copy_state_from_device(&ctx, &snap).map_err(|e| e.to_string())?;
            }
            let timing = r
                .step_profiled_no_readback(
                    &ctx,
                    parameters(n),
                    ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
                    1.0,
                    b.host(),
                )
                .map_err(|e| e.to_string())?;
            total_kernel_ms += timing.total_ms as f64;
            consumed.push(json!({"step":n,"batch_sha256":digest_json(b)}));
            if checkpoint {
                eprintln!("issue13 gate1 arm={arm} step={n}");
                let weights = r.read_weights(&ctx).map_err(|e| e.to_string())?;
                let optimizer = r.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
                let temp = arm_dir.join(format!(".{n}.partial"));
                if temp.exists() {
                    fs::remove_dir_all(&temp).map_err(|e| e.to_string())?;
                }
                fs::create_dir(&temp).map_err(|e| e.to_string())?;
                super::write_cuda_cpp_sfnn_weights_bin(
                    &temp.join("state.bin"),
                    &weights,
                    PostPairwiseTransform::Identity,
                    &optimizer,
                    None,
                    n,
                    n,
                )?;
                fs::File::open(temp.join("state.bin")).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
                let batch = SfnnForwardDeviceBatch::from_host(
                    &ctx,
                    SfnnForwardHostBatch {
                        stm_indices: &b.stm,
                        nstm_indices: &b.nstm,
                        buckets: &b.buckets,
                        batch_size: b.size(),
                        max_active: b.max_active,
                    },
                )
                .map_err(|e| e.to_string())?;
                let workspace = SfnnForwardWorkspace::new(&ctx, SfnnForwardWorkspaceLayout::new(shape, b.size()))
                    .map_err(|e| e.to_string())?;
                r.forward_current_weights(&ctx, &batch, &workspace).map_err(|e| e.to_string())?;
                let trace = workspace.download(&ctx).map_err(|e| e.to_string())?;
                let quantized = quantized_observation(args, &weights, shape, b, &temp.join("nn.bin"), &trace.output)?;
                let before = before.as_ref().unwrap();
                let mut updates = serde_json::Map::new();
                for (name, after) in weights_map(&weights) {
                    let prev = weights_map(before)[&name];
                    let delta: Vec<f32> = after.iter().zip(prev).map(|(a, b)| a - b).collect();
                    updates.insert(name, stats(&delta)?);
                }
                let row = json!({"spec_sha256":digest_json(spec),"step":n,"arm":arm,"resolved_path":format!("{:?}",r.l0_backward_path()),"initial_state_sha256":digest_json(&identity),"initial_state":identity,"batch_sequence_sha256":digest_json(&consumed),"consumed_batches":consumed,"state_sha256":file_hash(&temp.join("state.bin"))?,"weights":tensor_stats(weights_map(&weights))?,"gradients":gradient_stats,"updates":updates,"post_pairwise":stats(&trace.combined)?,"float_output":stats(&trace.output)?,"float_outputs":trace.output,"quantized":quantized,"timing":{"checkpoint_step_gpu_ms":timing.total_ms,"forward_ms":timing.forward_ms,"backward_ms":timing.backward_ms,"l0_ms":timing.backward_stages.l0_ms,"pairwise_kernel_ms":timing.backward_stages.pairwise_ms,"sparse_l0_kernel_ms":timing.backward_stages.sparse_l0_ms,"l1_ms":timing.backward_stages.l1_ms,"l2_ms":timing.backward_stages.l2_ms,"l3_ms":timing.backward_stages.l3_ms,"update_ms":timing.update_ms,"total_gpu_ms":total_kernel_ms,"wall_seconds":arm_started.elapsed().as_secs_f64(),"profiled_positions_per_second":(n*b.size()) as f64/(total_kernel_ms/1000.0)},"resume":resume_identity(n,seq.len()),"clean_completion":true});
                publish(&temp.join("metadata.json"), &row)?;
                fs::rename(&temp, arm_dir.join(n.to_string())).map_err(|e| e.to_string())?;
                saved.insert(n.to_string(), row);
            }
        }
        arm_reports.insert(arm.into(), json!({"initial_state":identity,"checkpoints":saved,"clean_completion":true}));
    }
    for n in checkpoints {
        let a_dir = spec.output.join("D-A").join(n.to_string());
        let b_dir = spec.output.join("D-A-prime").join(n.to_string());
        let a_row = &arm_reports["D-A"]["checkpoints"][n.to_string()];
        let b_row = &arm_reports["D-A-prime"]["checkpoints"][n.to_string()];
        if a_row["batch_sequence_sha256"] != b_row["batch_sequence_sha256"]
            || a_row["initial_state_sha256"] != b_row["initial_state_sha256"]
        {
            return Err("DATA_INTEGRITY_STOP: paired checkpoint identity differs".into());
        }
        let a = super::load_cuda_cpp_sfnn_initial_state(
            &a_dir.join("state.bin"),
            args,
            super::CudaCppSfnnFeatureKind::Halfka2,
        )?;
        let b = super::load_cuda_cpp_sfnn_initial_state(
            &b_dir.join("state.bin"),
            args,
            super::CudaCppSfnnFeatureKind::Halfka2,
        )?;
        let aw = super::sfnn_initial_weights_into_readback(a.weights);
        let bw = super::sfnn_initial_weights_into_readback(b.weights);
        let p = compare_tolerance(weights_map(&aw), weights_map(&bw), 1e-5, 1e-3)?;
        let o = compare_tolerance(
            optimizer_host_map(a.optimizer_states.as_ref().ok_or("missing optimizer")?.as_host()),
            optimizer_host_map(b.optimizer_states.as_ref().ok_or("missing optimizer")?.as_host()),
            1e-5,
            1e-3,
        )?;
        let floats = |row: &Value, key: &str| -> Vec<f32> {
            row[key].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect()
        };
        let fa = floats(a_row, "float_outputs");
        let fb = floats(b_row, "float_outputs");
        let f = metric_tolerance(&fa, &fb, vec![16], 1e-5, 1e-3)?;
        let qa = floats(&a_row["quantized"], "outputs");
        let qb = floats(&b_row["quantized"], "outputs");
        let q = metric_tolerance(&qa, &qb, vec![16], 0.01, 1e-3)?;
        let ft_nonzero_delta = (a_row["quantized"]["ft_weight_nonzero_fraction"].as_f64().unwrap()
            - b_row["quantized"]["ft_weight_nonzero_fraction"].as_f64().unwrap())
        .abs();
        let passed = all_passed(&p) && all_passed(&o) && f["passed"] == true && ft_nonzero_delta <= 1e-5;
        comparisons.insert(n.to_string(),json!({"parameters":p,"optimizer":o,"float_output":f,"quantized_output_descriptive":q,"ft_quantized_nonzero_fraction_abs_delta":ft_nonzero_delta,"float_sign_agreement":fa.iter().zip(&fb).filter(|(a,b)|a.signum()==b.signum()).count() as f64/16.0,"passed":passed}));
    }
    let passed = comparisons.values().all(|v| v["passed"] == true);
    Ok(
        json!({"schema_version":"issue13-gate1-v2","passed":passed,"typed_stop":if passed{Value::Null}else{json!("IMPLEMENTATION_STOP")},"absolute_tolerance":1e-5,"relative_tolerance":1e-3,"arms":arm_reports,"comparisons":comparisons,"conclusion":if passed{"tested-horizon-no-cuda-path-degradation"}else{"paired-trajectory-diverged-investigate-implementation"},"limitations":["16 positions per step, fixed 64-record sequence","1000-step horizon only","profile timing is diagnostic, not a performance benchmark"]}),
    )
}

pub fn run(args: &super::Args, path: &Path) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|e| format!("DATA_INTEGRITY_STOP: {e}"))?;
    let spec: Spec = serde_json::from_slice(&bytes).map_err(|e| format!("CONFIGURATION_STOP: {e}"))?;
    if spec.sealed_test {
        return Err("SEALED_POLICY_STOP: qualification cannot use sealed test".into());
    }
    if spec.schema_version != "issue13-qualification-v2"
        || !["gate0", "gate1"].contains(&spec.stage.as_str())
        || spec.seed != SEED
        || spec.repetitions != 2
        || spec.absolute_tolerance != ATOL
        || spec.relative_tolerance != RTOL
        || spec.fixtures != FIXTURES
        || args.arch().cli_name() != "SFNN_halfka2_1024_7_64_k3k3"
        || !super::effective_sfnn_factorizer_spec(args).shared
    {
        return Err("CONFIGURATION_STOP: qualification contract mismatch".into());
    }
    if file_hash(&spec.teacher)? != spec.teacher_sha256
        || file_hash(&std::env::current_exe().map_err(|e| e.to_string())?)? != spec.binary_sha256
    {
        return Err("DATA_INTEGRITY_STOP: preregistered input/binary differs".into());
    }
    let fixed_args = super::Args::try_parse_from([
        "bulletou",
        "--arch",
        "SFNN_halfka2_1024_7_64_k3k3",
        "--backend",
        "cuda-cpp",
        "--teacher",
        spec.teacher.to_str().ok_or("invalid teacher path")?,
        "--cuda-cpp-train-steps",
        "1",
        "--sfnn-factorizer",
        "shared",
    ])
    .map_err(|e| format!("CONFIGURATION_STOP: {e}"))?;
    let args = &fixed_args;
    if spec.output.exists() {
        let dispatch: Value = serde_json::from_slice(
            &fs::read(spec.output.join("DISPATCH.json")).map_err(|e| format!("DATA_INTEGRITY_STOP: {e}"))?,
        )
        .map_err(|e| e.to_string())?;
        if dispatch["preregistration_sha256"] != format!("{:x}", Sha256::digest(&bytes)) {
            return Err("DATA_INTEGRITY_STOP: resumed preregistration differs".into());
        }
        if spec.output.join("REPORT.json").exists() {
            return Err("CONFIGURATION_STOP: qualification already completed; use its report".into());
        }
    } else {
        fs::create_dir_all(&spec.output).map_err(|e| format!("RESOURCE_STOP: {e}"))?;
        publish(
            &spec.output.join("DISPATCH.json"),
            &json!({"preregistration_sha256":format!("{:x}",Sha256::digest(&bytes)),"config":spec,"args":std::env::args().collect::<Vec<_>>(),"started_unix_seconds":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()}),
        )?;
    }
    let result = if spec.stage == "gate0" { gate0(args, &spec) } else { gate1(args, &spec) };
    let mut report = match result {
        Ok(v) => v,
        Err(e) => {
            json!({"schema_version":if spec.stage=="gate0"{"issue13-gate0-v2"}else{"issue13-gate1-v2"},"passed":false,"typed_stop":if e.contains("RESOURCE_STOP") || e.to_lowercase().contains("out of memory") || e.to_lowercase().contains("memory allocation"){"RESOURCE_STOP"}else if e.contains("DATA_INTEGRITY_STOP"){"DATA_INTEGRITY_STOP"}else{"IMPLEMENTATION_STOP"},"error":e})
        }
    };
    report["preregistration_sha256"] = json!(format!("{:x}", Sha256::digest(&bytes)));
    report["provenance"] = json!(spec.provenance);
    report["binary_sha256"] = json!(spec.binary_sha256);
    report["completed_unix_seconds"] =
        json!(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs());
    report["resolved_config"] = json!({"architecture":args.arch().cli_name(),"backend":"cuda-cpp","factorizer":"shared","device":0,"batch_size":16,"seed":SEED,"initialization":{"scale":args.nnue_pytorch_init_scale,"bias":args.sfnn_init_bias.cli_name(),"l2_scale":super::effective_sfnn_init_l2_scale(args),"l3_scale":super::effective_sfnn_init_l3_scale(args)},"optimizer":{"name":"Ranger","lr":0.000875,"beta1":0.9,"beta2":0.999,"epsilon":1e-8,"n_sma_threshold":5,"lookahead_alpha":0.5,"lookahead_period":6,"decay":0,"gradient_factor":1},"loss":{"name":"SigmoidPow","exponent":2,"target":"sigmoid(score/600)"}});
    let report_name = if report["typed_stop"] == "RESOURCE_STOP" {
        format!("RESOURCE_STOP-{}.json", std::process::id())
    } else {
        "REPORT.json".into()
    };
    publish(&spec.output.join(report_name), &report)?;
    if report["passed"] != true {
        return Err(format!("{}: qualification did not pass; see REPORT.json", report["typed_stop"]));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resume_rejects_changed_scheduler_rng_and_cursor() {
        let original = resume_identity(10, 4);
        assert!(validate_resume_identity(&original, 10, 4).is_ok());
        for (field, value) in [
            ("lr", json!(0.1)),
            ("rng_seed", json!(0)),
            ("runtime_rng_draws", json!(1)),
            ("next_batch", json!(3)),
            ("optimizer_steps", json!(9)),
            ("pending_gradients", json!(true)),
        ] {
            let mut changed = original.clone();
            changed[field] = value;
            assert!(validate_resume_identity(&changed, 10, 4).unwrap_err().contains("DATA_INTEGRITY_STOP"));
        }
    }
    #[test]
    fn sealed_registration_stops_before_input_or_cuda_access() {
        let path = std::env::temp_dir().join(format!("issue13-sealed-{}.json", std::process::id()));
        let value = json!({"schema_version":"issue13-qualification-v2","stage":"gate0",
            "teacher":"nonexistent-input","teacher_sha256":"invalid","binary_sha256":"invalid",
            "output":"unused-output","seed":SEED,"repetitions":2,"absolute_tolerance":ATOL,
            "relative_tolerance":RTOL,"fixtures":FIXTURES,"sealed_test":true,"provenance":{}});
        let mut file = fs::OpenOptions::new().write(true).create_new(true).open(&path).unwrap();
        file.write_all(&serde_json::to_vec(&value).unwrap()).unwrap();
        drop(file);
        let args = crate::Args::try_parse_from([
            "bulletou",
            "--arch",
            "SFNN_halfka2_1024_7_64_k3k3",
            "--backend",
            "cuda-cpp",
            "--teacher",
            "nonexistent-input",
            "--cuda-cpp-train-steps",
            "1",
        ])
        .unwrap();
        let error = run(&args, &path).unwrap_err();
        fs::remove_file(path).unwrap();
        assert!(error.starts_with("SEALED_POLICY_STOP"), "{error}");
    }
    #[test]
    #[ignore = "requires CUDA and explicit ISSUE13_ROUNDTRIP_STATE checkpoint"]
    fn production_checkpoint_cuda_roundtrip_is_bitwise() {
        let source = PathBuf::from(std::env::var_os("ISSUE13_ROUNDTRIP_STATE").expect("checkpoint required"));
        let expected = file_hash(&source).unwrap();
        let args = crate::Args::try_parse_from([
            "bulletou",
            "--arch",
            "SFNN_halfka2_1024_7_64_k3k3",
            "--backend",
            "cuda-cpp",
            "--teacher",
            "unused",
            "--cuda-cpp-train-steps",
            "1",
            "--sfnn-factorizer",
            "shared",
        ])
        .unwrap();
        let loaded =
            crate::load_cuda_cpp_sfnn_initial_state(&source, &args, crate::CudaCppSfnnFeatureKind::Halfka2).unwrap();
        let n = loaded.completed_steps;
        assert_eq!(n, 10, "this check targets the planned step-10 restart");
        assert_eq!(loaded.optimizer_steps, n);
        let ctx = Context::new(0).unwrap();
        let batch = Batch {
            stm: vec![-1; 16],
            nstm: vec![-1; 16],
            buckets: vec![0; 16],
            targets: vec![0.5; 16],
            weights: vec![1.0; 16],
            max_active: 1,
        };
        let mut r = runner(&ctx, loaded.weights.as_host(), &batch, SfnnL0BackwardSelector::Auto).unwrap();
        r.upload_state_from_host(
            &ctx,
            loaded.weights.as_host(),
            loaded.optimizer_states.as_ref().unwrap().as_host(),
            SfnnFactorizerActive { shared: true, ..SfnnFactorizerActive::NONE },
            SfnnFactorizerAlpha::ONE,
        )
        .unwrap();
        drop(loaded);
        let w = r.read_weights(&ctx).unwrap();
        let o = r.read_optimizer_states(&ctx).unwrap();
        drop(r);
        let target = std::env::temp_dir().join(format!("issue13-roundtrip-{}.bin", std::process::id()));
        assert!(!target.exists());
        crate::write_cuda_cpp_sfnn_weights_bin(&target, &w, PostPairwiseTransform::Identity, &o, None, n, n).unwrap();
        let actual = file_hash(&target).unwrap();
        fs::remove_file(target).unwrap();
        println!(
            "{}",
            json!({"check":"checkpoint-cuda-roundtrip","source_sha256":expected,
            "roundtrip_sha256":actual,"updates":0,"passed":expected==actual})
        );
        assert_eq!(expected, actual, "disk loader/GPU upload/readback/state writer changed checkpoint bytes");
    }
    #[test]
    fn duplicate_feature_rejected() {
        let b = Batch {
            stm: vec![1, 1],
            nstm: vec![2, -1],
            buckets: vec![0],
            targets: vec![0.5],
            weights: vec![1.0],
            max_active: 2,
        };
        assert!(b.validate().is_err());
    }
    #[test]
    fn acceptance_recomputes_metric_decision() {
        let mut m = metric(&[0.0], &[0.1], vec![1]).unwrap();
        m["passed"] = json!(true);
        m["violations"] = json!(0);
        assert!(!all_passed(&json!({"tensor":m})));
        assert!(!all_passed(&json!({"tensor":{"passed":true}})));
    }
    #[test]
    fn metric_cannot_hide_sparse_large_error_in_rms() {
        let a = vec![0.0; 10000];
        let mut b = a.clone();
        b[9] = 0.01;
        let m = metric(&a, &b, vec![10000]).unwrap();
        assert_eq!(m["passed"], false);
        assert_eq!(m["violations"], 1);
        assert!(metric(&[f32::NAN], &[0.0], vec![1]).is_err());
    }
}
