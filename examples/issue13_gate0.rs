use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;

use bulletou_cuda_cpp::{
    Context, RAdamUpdateParams, RangerParamStateReadback, RangerUpdateParams, ScalarLossKind, SfnnBackwardReadback,
    SfnnFactorizerActive, SfnnFactorizerAlpha, SfnnForwardHostWeights, SfnnForwardReadback, SfnnForwardShape,
    SfnnL0BackwardSelector, SfnnRangerOptimizerStatesReadback, SfnnTrainStepHostBatch, SfnnTrainStepRunner,
    SfnnTrainWeightsReadback,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const INPUT_SIZE: usize = 133_578;
const BASE_INPUT_SIZE: usize = 131_949;
const PIECE_INPUTS: usize = 1_629;
const FT: usize = 1_024;

#[derive(Default)]
struct Metric {
    count: u64,
    max_abs: f64,
    max_rel: f64,
    square_error: f64,
    left_square: f64,
    right_square: f64,
    dot: f64,
    worst_tensor: String,
}

impl Metric {
    fn add(&mut self, left: &[f32], right: &[f32]) -> Result<(), String> {
        self.add_named("unnamed", left, right)
    }

    fn add_named(&mut self, name: &str, left: &[f32], right: &[f32]) -> Result<(), String> {
        if left.len() != right.len() {
            return Err(format!("metric length mismatch: {} != {}", left.len(), right.len()));
        }
        for (&a, &b) in left.iter().zip(right) {
            if !a.is_finite() || !b.is_finite() {
                return Err("non-finite Gate 0 tensor value".to_string());
            }
            let a = f64::from(a);
            let b = f64::from(b);
            let d = (a - b).abs();
            self.count += 1;
            if d > self.max_abs {
                self.max_abs = d;
                self.worst_tensor = name.to_string();
            }
            self.max_rel = self.max_rel.max(d / a.abs().max(b.abs()).max(1.0e-30));
            self.square_error += d * d;
            self.left_square += a * a;
            self.right_square += b * b;
            self.dot += a * b;
        }
        Ok(())
    }

    fn value(&self) -> Value {
        let rms = if self.count == 0 { 0.0 } else { (self.square_error / self.count as f64).sqrt() };
        let relative_l2 = self.square_error.sqrt() / self.left_square.sqrt().max(self.right_square.sqrt()).max(1.0e-30);
        let cosine = if self.left_square == 0.0 && self.right_square == 0.0 {
            1.0
        } else {
            self.dot / (self.left_square * self.right_square).sqrt().max(1.0e-30)
        };
        json!({
            "max_abs": self.max_abs,
            "max_rel": self.max_rel,
            "rms": rms,
            "relative_l2": relative_l2,
            "cosine": cosine,
        })
    }

    fn worst_tensor(&self) -> &str {
        &self.worst_tensor
    }
}

struct OwnedWeights {
    l0w: Vec<f32>,
    l0b: Vec<f32>,
    l1w: Vec<f32>,
    l1b: Vec<f32>,
    l1fw: Vec<f32>,
    l1fb: Vec<f32>,
    l2w: Vec<f32>,
    l2b: Vec<f32>,
    l2fw: Vec<f32>,
    l2fb: Vec<f32>,
    l3w: Vec<f32>,
    l3b: Vec<f32>,
    l3fw: Vec<f32>,
    l3fb: Vec<f32>,
}

impl OwnedWeights {
    fn host(&self) -> SfnnForwardHostWeights<'_> {
        SfnnForwardHostWeights {
            shape: production_shape(),
            l0w: &self.l0w,
            l0b: &self.l0b,
            l1w: &self.l1w,
            l1b: &self.l1b,
            l1fw: Some(&self.l1fw),
            l1fb: Some(&self.l1fb),
            l1axw: None,
            l1axb: None,
            l2w: &self.l2w,
            l2b: &self.l2b,
            l2fw: Some(&self.l2fw),
            l2fb: Some(&self.l2fb),
            l2axw: None,
            l2axb: None,
            l3w: &self.l3w,
            l3b: &self.l3b,
            l3fw: Some(&self.l3fw),
            l3fb: Some(&self.l3fb),
            l3axw: None,
            l3axb: None,
        }
    }
}

fn production_shape() -> SfnnForwardShape {
    SfnnForwardShape {
        post_pairwise_transform: bulletou_cuda_cpp::PostPairwiseTransform::Identity,
        input_size: INPUT_SIZE,
        ft_size: FT,
        l1_hidden: 7,
        l1_skip: true,
        l2_size: 64,
        num_stacks: 9,
        l1_group_count: 1,
        l1_common_size: 0,
        l1_shard_size: 0,
        factorizer_king_axis_dim: 3,
        factorizer_hand_axis_dim: 0,
        factorizer_progress_axis: false,
        factorizer_king_hand_pair: false,
        factorizer_king_progress_pair: false,
        factorizer_hand_progress_pair: false,
    }
}

fn patterned(len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|i| ((i % 29) as f32 - 14.0) * scale).collect()
}

fn make_weights(indices: &[i32]) -> OwnedWeights {
    let shape = production_shape();
    let mut l0w = vec![0.0; shape.input_size * shape.ft_size];
    for &feature in indices {
        if feature < 0 || feature as usize >= BASE_INPUT_SIZE {
            continue;
        }
        for mapped in [feature as usize, BASE_INPUT_SIZE + feature as usize % PIECE_INPUTS] {
            let base = mapped * FT;
            for row in 0..FT {
                l0w[base + row] = (((feature as usize + row) % 31) as f32 - 15.0) * 0.000_031_25;
            }
        }
    }
    let eps = f32::EPSILON * 4.0;
    let boundary = [0.0, eps, 63.0 / 128.0 - eps, 63.0 / 128.0, 63.0 / 128.0 + eps, 1.0 - eps, 1.0, 1.0 + eps];
    let l0b = (0..FT).map(|i| boundary[i % boundary.len()]).collect();
    OwnedWeights {
        l0w,
        l0b,
        l1w: patterned(shape.l1w_len().unwrap(), 0.000_02),
        l1b: patterned(shape.num_stacks * shape.l1_out(), 0.000_1),
        l1fw: patterned(shape.ft_size * shape.l1_out(), 0.000_01),
        l1fb: patterned(shape.l1_out(), 0.000_1),
        l2w: patterned(shape.num_stacks * shape.l2_size * shape.l2_in(), 0.000_2),
        l2b: patterned(shape.num_stacks * shape.l2_size, 0.000_1),
        l2fw: patterned(shape.l2_size * shape.l2_in(), 0.000_1),
        l2fb: patterned(shape.l2_size, 0.000_1),
        l3w: patterned(shape.num_stacks * shape.l2_size, 0.000_5),
        l3b: patterned(shape.num_stacks, 0.000_1),
        l3fw: patterned(shape.l2_size, 0.000_2),
        l3fb: vec![0.000_1],
    }
}

struct FixtureBatch {
    stm: Vec<i32>,
    nstm: Vec<i32>,
    buckets: Vec<i32>,
    targets: Vec<f32>,
    weights: Vec<f32>,
    batch_size: usize,
    max_active: usize,
}

fn parse_batch(root: &Value) -> Result<FixtureBatch, String> {
    let batch = root.get("batch").and_then(Value::as_object).ok_or("fixture batch is missing")?;
    let ints = |name: &str| -> Result<Vec<i32>, String> {
        batch
            .get(name)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("missing {name}"))?
            .iter()
            .map(|v| v.as_i64().and_then(|x| i32::try_from(x).ok()).ok_or_else(|| format!("invalid {name}")))
            .collect()
    };
    let floats = |name: &str| -> Result<Vec<f32>, String> {
        batch
            .get(name)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("missing {name}"))?
            .iter()
            .map(|v| v.as_f64().map(|x| x as f32).filter(|x| x.is_finite()).ok_or_else(|| format!("invalid {name}")))
            .collect()
    };
    let batch_size = batch.get("batch_size").and_then(Value::as_u64).ok_or("missing batch_size")? as usize;
    let max_active = batch.get("max_active").and_then(Value::as_u64).ok_or("missing max_active")? as usize;
    let out = FixtureBatch {
        stm: ints("stm")?,
        nstm: ints("nstm")?,
        buckets: ints("buckets")?,
        targets: floats("targets")?,
        weights: floats("weights")?,
        batch_size,
        max_active,
    };
    if out.stm.len() != batch_size * max_active
        || out.nstm.len() != batch_size * max_active
        || out.buckets.len() != batch_size
        || out.targets.len() != batch_size
        || out.weights.len() != batch_size
    {
        return Err("fixture batch dimensions mismatch".to_string());
    }
    Ok(out)
}

fn host_batch(batch: &FixtureBatch) -> SfnnTrainStepHostBatch<'_> {
    SfnnTrainStepHostBatch {
        stm_indices: &batch.stm,
        nstm_indices: &batch.nstm,
        buckets: &batch.buckets,
        targets: &batch.targets,
        entry_weights: &batch.weights,
        batch_size: batch.batch_size,
        max_active: batch.max_active,
    }
}

fn new_runner<'a>(
    ctx: &Context,
    weights: SfnnForwardHostWeights<'a>,
    batch: &FixtureBatch,
    selector: SfnnL0BackwardSelector,
) -> Result<SfnnTrainStepRunner, String> {
    SfnnTrainStepRunner::new_with_factorizer_and_l0_backward_selector(
        ctx,
        weights,
        batch.batch_size,
        batch.max_active,
        SfnnFactorizerActive { shared: true, ..SfnnFactorizerActive::NONE },
        SfnnFactorizerAlpha::ONE,
        selector,
    )
    .map_err(|e| e.to_string())
}

fn params() -> RangerUpdateParams {
    RangerUpdateParams {
        radam: RAdamUpdateParams {
            step: 1,
            learning_rate: 0.000875,
            min_weight: -1.98,
            max_weight: 1.98,
            ..RAdamUpdateParams::default()
        },
        lookahead_alpha: 0.5,
        lookahead_period: 6,
    }
}

fn add_forward(metric: &mut Metric, a: &SfnnForwardReadback, b: &SfnnForwardReadback) -> Result<(), String> {
    for (x, y) in [
        (&a.stm_l0, &b.stm_l0),
        (&a.nstm_l0, &b.nstm_l0),
        (&a.combined, &b.combined),
        (&a.l1, &b.l1),
        (&a.l2_input, &b.l2_input),
        (&a.l2, &b.l2),
        (&a.output, &b.output),
    ] {
        metric.add(x, y)?;
    }
    Ok(())
}

fn add_backward(metric: &mut Metric, a: &SfnnBackwardReadback, b: &SfnnBackwardReadback) -> Result<(), String> {
    macro_rules! add { ($($field:ident),+ $(,)?) => { $(metric.add_named(stringify!($field), &a.$field, &b.$field)?;)+ }; }
    add!(
        l0w_gradients,
        l0b_gradients,
        l1w_gradients,
        l1b_gradients,
        l1fw_gradients,
        l1fb_gradients,
        l1axw_gradients,
        l1axb_gradients,
        l2w_gradients,
        l2b_gradients,
        l2fw_gradients,
        l2fb_gradients,
        l2axw_gradients,
        l2axb_gradients,
        l3w_gradients,
        l3b_gradients,
        l3fw_gradients,
        l3fb_gradients,
        l3axw_gradients,
        l3axb_gradients
    );
    Ok(())
}

fn add_weights(metric: &mut Metric, a: &SfnnTrainWeightsReadback, b: &SfnnTrainWeightsReadback) -> Result<(), String> {
    macro_rules! add { ($($field:ident),+ $(,)?) => { $(metric.add_named(stringify!($field), &a.$field, &b.$field)?;)+ }; }
    add!(l0w, l0b, l1w, l1b, l2w, l2b, l3w, l3b);
    for (x, y) in [
        (&a.l1fw, &b.l1fw),
        (&a.l1fb, &b.l1fb),
        (&a.l1axw, &b.l1axw),
        (&a.l1axb, &b.l1axb),
        (&a.l2fw, &b.l2fw),
        (&a.l2fb, &b.l2fb),
        (&a.l2axw, &b.l2axw),
        (&a.l2axb, &b.l2axb),
        (&a.l3fw, &b.l3fw),
        (&a.l3fb, &b.l3fb),
        (&a.l3axw, &b.l3axw),
        (&a.l3axb, &b.l3axb),
    ] {
        match (x, y) {
            (Some(x), Some(y)) => metric.add(x, y)?,
            (None, None) => {}
            _ => return Err("optional weight mismatch".to_string()),
        }
    }
    Ok(())
}

fn add_ranger(metric: &mut Metric, a: &RangerParamStateReadback, b: &RangerParamStateReadback) -> Result<(), String> {
    metric.add(&a.momentum, &b.momentum)?;
    metric.add(&a.velocity, &b.velocity)?;
    metric.add(&a.slow_params, &b.slow_params)
}
fn add_optimizer(
    metric: &mut Metric,
    a: &SfnnRangerOptimizerStatesReadback,
    b: &SfnnRangerOptimizerStatesReadback,
) -> Result<(), String> {
    macro_rules! add { ($($field:ident),+ $(,)?) => { $(add_ranger(metric,&a.$field,&b.$field)?;)+ }; }
    add!(l0w, l0b, l1w, l1b, l2w, l2b, l3w, l3b);
    for (x, y) in [
        (&a.l1fw, &b.l1fw),
        (&a.l1fb, &b.l1fb),
        (&a.l1axw, &b.l1axw),
        (&a.l1axb, &b.l1axb),
        (&a.l2fw, &b.l2fw),
        (&a.l2fb, &b.l2fb),
        (&a.l2axw, &b.l2axw),
        (&a.l2axb, &b.l2axb),
        (&a.l3fw, &b.l3fw),
        (&a.l3fb, &b.l3fb),
        (&a.l3axw, &b.l3axw),
        (&a.l3axb, &b.l3axb),
    ] {
        match (x, y) {
            (Some(x), Some(y)) => add_ranger(metric, x, y)?,
            (None, None) => {}
            _ => return Err("optional optimizer mismatch".to_string()),
        }
    }
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn path_name(path: bulletou_cuda_cpp::SfnnL0BackwardPath) -> &'static str {
    match path {
        bulletou_cuda_cpp::SfnnL0BackwardPath::InverseIndex => "inverse-index",
        bulletou_cuda_cpp::SfnnL0BackwardPath::MaterializedSparse => "materialized-sparse",
    }
}

fn publish_report(report_path: &Path, report: &Value) -> Result<(), String> {
    let parent = report_path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let nonce =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|e| e.to_string())?.as_nanos();
    let temporary = parent.join(format!(".issue13-report-{}-{nonce}.tmp", std::process::id()));
    let mut output = fs::OpenOptions::new().write(true).create_new(true).open(&temporary).map_err(|e| e.to_string())?;
    let result = (|| {
        let bytes = serde_json::to_vec_pretty(report).map_err(|e| e.to_string())?;
        output.write_all(&bytes).map_err(|e| e.to_string())?;
        output.sync_all().map_err(|e| e.to_string())?;
        drop(output);
        // Same-directory hard link publishes the complete file without replacing
        // existing evidence, including a concurrent writer's report.
        fs::hard_link(&temporary, report_path).map_err(|e| e.to_string())
    })();
    let cleanup = fs::remove_file(&temporary).map_err(|e| e.to_string());
    result.and(cleanup)
}

pub fn run(device: i32, fixture_path: &Path, preregistration_path: &Path, report_path: &Path) -> Result<(), String> {
    let fixture_bytes = fs::read(fixture_path).map_err(|e| format!("cannot read fixtures: {e}"))?;
    let preregistration_bytes =
        fs::read(preregistration_path).map_err(|e| format!("cannot read preregistration: {e}"))?;
    let fixture_root: Value = serde_json::from_slice(&fixture_bytes).map_err(|e| format!("invalid fixtures: {e}"))?;
    let fixture_names = fixture_root.get("fixtures").and_then(Value::as_object).ok_or("fixture names missing")?;
    let required = [
        "production_halfka2",
        "empty_padding",
        "high_collision",
        "stm_nstm_active_count_mismatch",
        "float_boundaries",
        "integer_export_boundaries",
    ];
    if fixture_names.len() != required.len() || required.iter().any(|name| !fixture_names.contains_key(*name)) {
        return Err("Issue #13 fixture set mismatch".to_string());
    }
    let batch = parse_batch(&fixture_root)?;
    let all_indices = batch.stm.iter().chain(&batch.nstm).copied().collect::<Vec<_>>();
    let weights = make_weights(&all_indices);
    let mut initial_hash = Sha256::new();
    for values in [
        &weights.l0w,
        &weights.l0b,
        &weights.l1w,
        &weights.l1b,
        &weights.l1fw,
        &weights.l1fb,
        &weights.l2w,
        &weights.l2b,
        &weights.l2fw,
        &weights.l2fb,
        &weights.l3w,
        &weights.l3b,
        &weights.l3fw,
        &weights.l3fb,
    ] {
        for &v in values {
            initial_hash.update(v.to_le_bytes());
        }
    }
    let initial_state_sha256 = format!("{:x}", initial_hash.finalize());
    let ctx = Context::new(device).map_err(|e| e.to_string())?;

    let mut da = new_runner(&ctx, weights.host(), &batch, SfnnL0BackwardSelector::Auto)?;
    let da_path = path_name(da.l0_backward_path());
    da.step_no_readback_with_loss_finalize_and_update(
        &ctx,
        params(),
        ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
        1.0,
        host_batch(&batch),
        true,
        false,
    )
    .map_err(|e| e.to_string())?;
    let da_forward = da.read_forward_trace(&ctx).map_err(|e| e.to_string())?;
    let da_loss = da.read_loss(&ctx).map_err(|e| e.to_string())?;
    let da_backward = da.read_backward_trace(&ctx).map_err(|e| e.to_string())?;
    drop(da);

    let mut dap = new_runner(&ctx, weights.host(), &batch, SfnnL0BackwardSelector::MaterializedSparse)?;
    let dap_path = path_name(dap.l0_backward_path());
    dap.fill_l0_gradient_sentinel(&ctx, 17.0, -23.0).map_err(|e| e.to_string())?;
    dap.step_no_readback_with_loss_finalize_and_update(
        &ctx,
        params(),
        ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
        1.0,
        host_batch(&batch),
        true,
        false,
    )
    .map_err(|e| e.to_string())?;
    let dap_forward = dap.read_forward_trace(&ctx).map_err(|e| e.to_string())?;
    let dap_loss = dap.read_loss(&ctx).map_err(|e| e.to_string())?;
    let dap_backward = dap.read_backward_trace(&ctx).map_err(|e| e.to_string())?;
    let sentinel_replaced = !dap_backward.l0w_gradients.iter().any(|&v| v == 17.0)
        && !dap_backward.l0b_gradients.iter().any(|&v| v == -23.0);
    drop(dap);

    let mut output = Metric::default();
    add_forward(&mut output, &da_forward, &dap_forward)?;
    output.add(&da_loss.per_sample, &dap_loss.per_sample)?;
    output.add(&da_loss.mean_output_gradients, &dap_loss.mean_output_gradients)?;
    output.add(&[da_loss.weighted_sum, da_loss.mean], &[dap_loss.weighted_sum, dap_loss.mean])?;
    let mut gradient = Metric::default();
    add_backward(&mut gradient, &da_backward, &dap_backward)?;

    let mut da_update = new_runner(&ctx, weights.host(), &batch, SfnnL0BackwardSelector::Auto)?;
    da_update
        .step_no_readback_with_loss_finalize_and_update(
            &ctx,
            params(),
            ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
            1.0,
            host_batch(&batch),
            true,
            true,
        )
        .map_err(|e| e.to_string())?;
    let da_weights = da_update.read_weights(&ctx).map_err(|e| e.to_string())?;
    let da_optimizer = da_update.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
    drop(da_update);
    let mut dap_update = new_runner(&ctx, weights.host(), &batch, SfnnL0BackwardSelector::MaterializedSparse)?;
    dap_update
        .step_no_readback_with_loss_finalize_and_update(
            &ctx,
            params(),
            ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
            1.0,
            host_batch(&batch),
            true,
            true,
        )
        .map_err(|e| e.to_string())?;
    let dap_weights = dap_update.read_weights(&ctx).map_err(|e| e.to_string())?;
    let dap_optimizer = dap_update.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
    let mut parameter = Metric::default();
    add_weights(&mut parameter, &da_weights, &dap_weights)?;
    let mut optimizer = Metric::default();
    add_optimizer(&mut optimizer, &da_optimizer, &dap_optimizer)?;

    let comparisons = json!({"output_abs": output.value(), "gradient_abs": gradient.value(), "parameter_abs": parameter.value(), "optimizer_abs": optimizer.value()});
    let worst_tensors = json!({"output":output.worst_tensor(),"gradient":gradient.worst_tensor(),"parameter":parameter.worst_tensor(),"optimizer":optimizer.worst_tensor()});
    let fixture_sha256 = sha256(&fixture_bytes);
    let batch_sha256 = sha256(serde_json::to_vec(fixture_root.get("batch").unwrap()).unwrap().as_slice());
    if da_path != "inverse-index" || dap_path != "materialized-sparse" {
        return Err("IMPLEMENTATION_STOP: resolved Gate 0 kernel path mismatch".to_string());
    }
    // This is one shared synthetic batch. Fixture labels are not independent
    // executions, and scalar arithmetic is not an exporter differential.
    let measurement = json!({
        "fixture_bundle_sha256": fixture_sha256,
        "batch_sha256": batch_sha256,
        "initial_weights_sha256": initial_state_sha256,
        "kernel_paths": {"D-A": da_path, "D-A-prime": dap_path},
        "sentinel": {"requested_l0w": 17.0, "requested_l0b": -23.0,
                     "sentinel_values_absent_after_backward": sentinel_replaced},
        "comparisons": comparisons,
        "worst_tensors": worst_tensors,
    });
    let current_exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let bulletou_root = current_exe
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .ok_or("cannot resolve BulletOu root from executable")?;
    let commit = Command::new("git")
        .current_dir(bulletou_root)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|v| v.status.success())
        .map(|v| String::from_utf8_lossy(&v.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let binary_sha256 = sha256(&fs::read(&current_exe).map_err(|e| e.to_string())?);
    let dirty = Command::new("git")
        .current_dir(bulletou_root)
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|v| v.status.success())
        .map(|v| !v.stdout.is_empty())
        .unwrap_or(true);
    let cuda = Command::new(
        option_env!("CUDA_PATH").map(|root| format!("{root}/bin/nvcc")).unwrap_or_else(|| "nvcc".to_string()),
    )
    .arg("--version")
    .output()
    .ok()
    .filter(|v| v.status.success())
    .map(|v| String::from_utf8_lossy(&v.stdout).trim().replace('\n', " | "))
    .unwrap_or_else(|| "runtime-version-unavailable".to_string());
    let report = json!({
        "schema_version":"issue13-da-prime-diagnostic-v1", "producer":"bulletou-issue13-diagnostic",
        "acceptance_qualified":false, "preregistration_applied":false,
        "preregistration_sha256":sha256(&preregistration_bytes), "typed_stop":Value::Null,
        "provenance":{"commit":commit,"dirty":dirty,"binary_sha256":binary_sha256,"cuda":cuda,"gpu":bulletou_cuda_cpp::device_name(device).map_err(|e| e.to_string())?,"seed":20260913},
        "shared_batch_measurement":measurement,
        "unverified":["per-fixture differential", "tensor shapes and per-tensor metrics",
                      "real-data fixture provenance", "float boundary gradients", "integer/export differential",
                      "multiple microbatches", "soft-progress", "complete initial train state", "Gate 1 trajectory"],
    });
    if !sentinel_replaced {
        return Err("IMPLEMENTATION_STOP: L0 sentinel was not replaced".to_string());
    }
    publish_report(report_path, &report)?;
    println!("Issue #13 diagnostic report (not Gate 0 acceptance) = {}", report_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_publish_preserves_existing_evidence_and_cleans_temporary_files() {
        let root = std::env::temp_dir().join(format!("issue13-publish-test-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let path = root.join("report.json");
        let first = json!({"measurement": 0.125});
        publish_report(&path, &first).unwrap();
        assert!(publish_report(&path, &json!({"measurement": 9.0})).is_err());
        assert_eq!(serde_json::from_slice::<Value>(&fs::read(&path).unwrap()).unwrap(), first);
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }
}
