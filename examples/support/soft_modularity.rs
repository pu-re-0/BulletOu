//! Training-only Stage-1 DRSM controller for SFNN H1=15/H2=64.
use std::fmt::Write as _;

pub const H1:usize=15;
pub const H2:usize=64;

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Mode { RandomPartition, Adaptive }

#[derive(Clone,Debug)]
pub struct Config {
    pub mode:Mode, pub seed:u64, pub total_steps:usize, pub refresh_steps:usize,
    pub warmup_fraction:f64, pub freeze_fraction:f64, pub relax_fraction:f64,
    pub patience_refreshes:usize, pub n_null:usize, pub gain_min:f64, pub gain_full:f64,
    pub g_target:f64, pub g_max:f64, pub beta_floor:f64, pub target_grad_ratio:f64,
    pub lambda_min:f64, pub lambda_max:f64, pub lambda_ema:f64,
}

impl Config {
    pub fn validate(&self)->Result<(),String>{
        if self.total_steps==0 || self.refresh_steps==0 || self.n_null==0 || self.patience_refreshes==0 { return Err("softmod step/null/patience counts must be positive".into()) }
        if !(0.0..=1.0).contains(&self.warmup_fraction) || !(self.warmup_fraction..=1.0).contains(&self.freeze_fraction)
            || !(self.freeze_fraction..=1.0).contains(&self.relax_fraction) { return Err("softmod fractions must satisfy 0 <= warmup <= freeze <= relax <= 1".into()) }
        if !(self.gain_min.is_finite() && self.gain_full>self.gain_min) { return Err("softmod gain_full must exceed finite gain_min".into()) }
        if !(0.0<=self.g_target && self.g_target<=self.g_max && self.g_max<1.0) { return Err("softmod requires 0 <= g_target <= g_max < 1".into()) }
        if !(self.beta_floor>=0.0 && self.target_grad_ratio>0.0 && self.lambda_min>0.0 && self.lambda_max>=self.lambda_min && (0.0..1.0).contains(&self.lambda_ema)) { return Err("invalid softmod loss/lambda controls".into()) }
        Ok(())
    }
}

#[derive(Clone,Debug)]
struct Partition { h1:[u8;H1], h2:[u8;H2], q:f64 }

#[derive(Clone,Debug)]
struct StackState { partition:Partition, q_null:f64, q_null_std:f64, gain:f64, confidence:f64, consecutive:usize, stability:f64 }

#[derive(Clone,Debug,PartialEq)]
pub struct DeviceUpdate { pub h1_labels:Vec<i32>, pub h2_labels:Vec<i32>, pub q_targets:Vec<f32>, pub q_floors:Vec<f32>, pub confidences:Vec<f32>, pub lambda_struct:f32, pub beta_floor:f32 }

#[derive(Clone,Debug)]
pub struct Controller { cfg:Config, stacks:Vec<StackState>, refresh_index:usize, lambda_ema:Option<f64>, frozen:bool, history:Vec<String> }

#[derive(Clone,Copy)] struct Rng(u64);
impl Rng { fn next(&mut self)->u64 { let mut x=self.0; x^=x<<13; x^=x>>7; x^=x<<17; self.0=x; x } fn shuffle<T>(&mut self,x:&mut[T]) { for i in (1..x.len()).rev(){ let j=(self.next()%(i as u64+1)) as usize; x.swap(i,j); } } }

fn exact(a:&[[f64;H1];H2])->Partition {
    let mut best=Partition{h1:[0;H1],h2:[0;H2],q:f64::INFINITY};
    let total:f64=a.iter().flatten().sum();
    for mask in 0u32..(1u32<<H1) { if mask.count_ones()!=8 {continue}
        let mut ranked=[(0.0,0usize);H2]; let mut c0=[0.0;H2]; let mut c1=[0.0;H2];
        for r in 0..H2 { let row_total:f64=a[r].iter().sum(); for h in 0..H1 {if mask&(1<<h)!=0 {c1[r]+=a[r][h]}} c0[r]=row_total-c1[r]; ranked[r]=(c0[r]-c1[r],r); }
        ranked.sort_by(|x,y|x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        let mut h2=[1u8;H2]; for x in &ranked[..32] {h2[x.1]=0}
        let off:f64=(0..H2).map(|r|if h2[r]==0{c0[r]}else{c1[r]}).sum(); let q=if total>0.0{off/total}else{0.0};
        let mut h1=[1u8;H1]; for (h,x) in h1.iter_mut().enumerate(){if mask&(1<<h)!=0{*x=0}}
        if q<best.q || (q==best.q && (h1,h2)<(best.h1,best.h2)) {best=Partition{h1,h2,q}}
    } best
}

fn mass(weights:&[f32],stack:usize)->[[f64;H1];H2] { let mut a=[[0.0;H1];H2]; let base=stack*H2*H1*2; for r in 0..H2 {for h in 0..H1 {a[r][h]=f64::from(weights[base+r*H1*2+h].abs()+weights[base+r*H1*2+H1+h].abs())}} a }
fn q_fixed(a:&[[f64;H1];H2],p:&Partition)->f64 {let mut t=0.0;let mut o=0.0;for r in 0..H2{for h in 0..H1{t+=a[r][h];if p.h1[h]!=p.h2[r]{o+=a[r][h]}}}if t>0.0{o/t}else{0.0}}
fn smoothstep(a:f64,b:f64,x:f64)->f64 {let t=((x-a)/(b-a)).clamp(0.0,1.0);t*t*(3.0-2.0*t)}

impl Controller {
    pub fn new(cfg:Config,num_stacks:usize)->Result<Self,String>{cfg.validate()?;if num_stacks==0{return Err("softmod needs stacks".into())}let zero=Partition{h1:[0;H1],h2:[0;H2],q:0.0};Ok(Self{cfg,stacks:(0..num_stacks).map(|_|StackState{partition:zero.clone(),q_null:0.0,q_null_std:0.0,gain:0.0,confidence:0.0,consecutive:0,stability:0.0}).collect(),refresh_index:0,lambda_ema:None,frozen:false,history:vec![]})}
    pub fn should_refresh(&self,step:usize)->bool {step==1 || step%self.cfg.refresh_steps==0 || (!self.frozen && step as f64/self.cfg.total_steps as f64>=self.cfg.freeze_fraction)}
    pub fn refresh(&mut self,step:usize,weights:&[f32],task_grad:&[f32])->Result<DeviceUpdate,String>{
        let expected=self.stacks.len()*H2*H1*2;if weights.len()!=expected||task_grad.len()!=expected{return Err(format!("softmod L2 shape mismatch: weights={} gradients={} expected={expected}",weights.len(),task_grad.len()))}
        let frac=step as f64/self.cfg.total_steps as f64; let can_change=!self.frozen && frac<self.cfg.freeze_fraction;
        for s in 0..self.stacks.len(){let a=mass(weights,s);let previous=self.stacks[s].partition.clone();let detected=exact(&a);
            if self.refresh_index==0 && self.cfg.mode==Mode::RandomPartition {let mut rng=Rng(self.cfg.seed^(s as u64+1).wrapping_mul(0x9e3779b97f4a7c15));let mut hi:Vec<usize>=(0..H1).collect();let mut ho:Vec<usize>=(0..H2).collect();rng.shuffle(&mut hi);rng.shuffle(&mut ho);let mut p=Partition{h1:[1;H1],h2:[1;H2],q:0.0};for&i in &hi[..8]{p.h1[i]=0}for&i in &ho[..32]{p.h2[i]=0}p.q=q_fixed(&a,&p);self.stacks[s].partition=p}
            else if self.cfg.mode==Mode::Adaptive && (self.refresh_index==0||can_change){self.stacks[s].partition=detected}
            else {self.stacks[s].partition.q=q_fixed(&a,&self.stacks[s].partition)}
            let mut qs=Vec::with_capacity(self.cfg.n_null);for trial in 0..self.cfg.n_null{let mut x=a;let mut rng=Rng(self.cfg.seed^(s as u64+1).wrapping_mul(0x9e3779b97f4a7c15)^(self.refresh_index as u64+1).wrapping_mul(0xbf58476d1ce4e5b9)^(trial as u64+1));for row in &mut x{rng.shuffle(row)}qs.push(exact(&x).q)}
            let mean=qs.iter().sum::<f64>()/qs.len() as f64;let var=qs.iter().map(|x|(x-mean)*(x-mean)).sum::<f64>()/(qs.len().saturating_sub(1).max(1) as f64);let gain=if mean>0.0{(mean-self.stacks[s].partition.q)/mean}else{0.0};
            let qualifies=self.cfg.mode==Mode::RandomPartition||gain>self.cfg.gain_min;self.stacks[s].consecutive=if qualifies{self.stacks[s].consecutive+1}else{0};let gate=if self.cfg.mode==Mode::RandomPartition{1.0}else{smoothstep(self.cfg.gain_min,self.cfg.gain_full,gain)};self.stacks[s].confidence=if frac>=self.cfg.warmup_fraction&&self.stacks[s].consecutive>=self.cfg.patience_refreshes{gate}else{0.0};self.stacks[s].q_null=mean;self.stacks[s].q_null_std=var.sqrt();self.stacks[s].gain=gain;
            self.stacks[s].stability=(previous.h1.iter().zip(self.stacks[s].partition.h1).filter(|(a,b)|**a==*b).count()+previous.h2.iter().zip(self.stacks[s].partition.h2).filter(|(a,b)|**a==*b).count()) as f64/(H1+H2) as f64;
        }
        if frac>=self.cfg.freeze_fraction{self.frozen=true}
        let mut update=self.device_update(frac,1.0);let struct_norm=structural_gradient_norm(weights,&update);let task_norm=task_grad.iter().map(|x|f64::from(*x)*f64::from(*x)).sum::<f64>().sqrt();if struct_norm>0.0&&task_norm>0.0{let raw=(self.cfg.target_grad_ratio*task_norm/struct_norm).clamp(self.cfg.lambda_min,self.cfg.lambda_max);self.lambda_ema=Some(self.lambda_ema.map_or(raw,|old|self.cfg.lambda_ema*old+(1.0-self.cfg.lambda_ema)*raw))}update=self.device_update(frac,self.lambda_ema.unwrap_or(self.cfg.lambda_min));for(s,st)in self.stacks.iter().enumerate(){let h1:Vec<_>=st.partition.h1.iter().copied().collect();let h2:Vec<_>=st.partition.h2.iter().copied().collect();let mut line=String::new();write!(line,"{{\"step\":{step},\"refresh\":{},\"stack\":{s},\"q_obs\":{},\"q_null_mean\":{},\"q_null_std\":{},\"gain\":{},\"confidence\":{},\"q_target\":{},\"q_floor\":{},\"lambda_struct\":{},\"partition_agreement_prev\":{},\"h1_labels\":{:?},\"h2_labels\":{:?},\"frozen\":{}}}",self.refresh_index,st.partition.q,st.q_null,st.q_null_std,st.gain,st.confidence,update.q_targets[s],update.q_floors[s],update.lambda_struct,st.stability,h1,h2,self.frozen).unwrap();self.history.push(line)}self.refresh_index+=1;Ok(update)
    }
    fn device_update(&self,frac:f64,lambda:f64)->DeviceUpdate{let relax=if frac<self.cfg.relax_fraction{1.0}else{1.0-0.5*((frac-self.cfg.relax_fraction)/(1.0-self.cfg.relax_fraction).max(1e-12)).clamp(0.0,1.0)};let mut u=DeviceUpdate{h1_labels:vec![],h2_labels:vec![],q_targets:vec![],q_floors:vec![],confidences:vec![],lambda_struct:(lambda*relax)as f32,beta_floor:self.cfg.beta_floor as f32};for s in &self.stacks{u.h1_labels.extend(s.partition.h1.map(i32::from));u.h2_labels.extend(s.partition.h2.map(i32::from));u.q_targets.push((s.q_null*(1.0-self.cfg.g_target))as f32);u.q_floors.push((s.q_null*(1.0-self.cfg.g_max))as f32);u.confidences.push(s.confidence as f32)}u}
    pub fn drain_history(&mut self)->impl Iterator<Item=String>+'_ {self.history.drain(..)}

    pub fn current_update(&self, step: usize) -> DeviceUpdate {
        self.device_update(step as f64 / self.cfg.total_steps as f64, self.lambda_ema.unwrap_or(self.cfg.lambda_min))
    }

    pub fn save_state(&self, path: &std::path::Path) -> Result<(), String> {
        let mut out = String::new();
        writeln!(
            out,
            "softmod-state-v1 {} {} {} {}",
            self.refresh_index,
            self.lambda_ema.map(f64::to_bits).unwrap_or(u64::MAX),
            u8::from(self.frozen),
            self.stacks.len()
        )
        .unwrap();
        for stack in &self.stacks {
            let h1: String = stack.partition.h1.iter().map(|x| char::from(b'0' + *x)).collect();
            let h2: String = stack.partition.h2.iter().map(|x| char::from(b'0' + *x)).collect();
            writeln!(
                out,
                "{} {} {} {} {} {} {} {} {}",
                stack.partition.q.to_bits(),
                stack.q_null.to_bits(),
                stack.q_null_std.to_bits(),
                stack.gain.to_bits(),
                stack.confidence.to_bits(),
                stack.consecutive,
                stack.stability.to_bits(),
                h1,
                h2
            )
            .unwrap();
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
        }
        std::fs::write(path, out).map_err(|e| format!("failed to write {}: {e}", path.display()))
    }

    pub fn load_state(cfg: Config, num_stacks: usize, path: &std::path::Path) -> Result<Self, String> {
        cfg.validate()?;
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        let mut lines = text.lines();
        let header: Vec<_> = lines.next().unwrap_or_default().split_whitespace().collect();
        if header.len() != 5 || header[0] != "softmod-state-v1" {
            return Err(format!("invalid softmod state header in {}", path.display()));
        }
        let parse_usize = |x: &str, label: &str| {
            x.parse::<usize>().map_err(|e| format!("invalid {label} in {}: {e}", path.display()))
        };
        let refresh_index = parse_usize(header[1], "refresh index")?;
        let lambda_bits = header[2]
            .parse::<u64>()
            .map_err(|e| format!("invalid lambda EMA in {}: {e}", path.display()))?;
        let frozen = header[3] == "1";
        let stored_stacks = parse_usize(header[4], "stack count")?;
        if stored_stacks != num_stacks {
            return Err(format!("softmod state has {stored_stacks} stacks, expected {num_stacks}"));
        }
        let mut stacks = Vec::with_capacity(num_stacks);
        for index in 0..num_stacks {
            let fields: Vec<_> = lines.next().unwrap_or_default().split_whitespace().collect();
            if fields.len() != 9 || fields[7].len() != H1 || fields[8].len() != H2 {
                return Err(format!("invalid softmod stack {index} in {}", path.display()));
            }
            let bits = |field: &str, label: &str| -> Result<f64, String> {
                field
                    .parse::<u64>()
                    .map(f64::from_bits)
                    .map_err(|e| format!("invalid {label} for stack {index}: {e}"))
            };
            let labels = |field: &str, len: usize, label: &str| -> Result<Vec<u8>, String> {
                let values: Vec<_> = field.bytes().map(|x| x.wrapping_sub(b'0')).collect();
                if values.len() != len || values.iter().any(|x| *x > 1) {
                    return Err(format!("invalid {label} for stack {index}"));
                }
                Ok(values)
            };
            let h1_vec = labels(fields[7], H1, "H1 labels")?;
            let h2_vec = labels(fields[8], H2, "H2 labels")?;
            let mut h1 = [0; H1];
            let mut h2 = [0; H2];
            h1.copy_from_slice(&h1_vec);
            h2.copy_from_slice(&h2_vec);
            stacks.push(StackState {
                partition: Partition { h1, h2, q: bits(fields[0], "q")? },
                q_null: bits(fields[1], "q_null")?,
                q_null_std: bits(fields[2], "q_null_std")?,
                gain: bits(fields[3], "gain")?,
                confidence: bits(fields[4], "confidence")?,
                consecutive: parse_usize(fields[5], "consecutive count")?,
                stability: bits(fields[6], "stability")?,
            });
        }
        if lines.any(|line| !line.trim().is_empty()) {
            return Err(format!("unexpected trailing data in {}", path.display()));
        }
        Ok(Self {
            cfg,
            stacks,
            refresh_index,
            lambda_ema: (lambda_bits != u64::MAX).then(|| f64::from_bits(lambda_bits)),
            frozen,
            history: vec![],
        })
    }
}

fn structural_gradient_norm(weights:&[f32],u:&DeviceUpdate)->f64 {let stacks=u.confidences.len();let mut ss=0.0;for s in 0..stacks{let base=s*H2*H1*2;let mut total=0.0;let mut off=0.0;for r in 0..H2{for c in 0..H1*2{let m=(f64::from(weights[base+r*H1*2+c]).powi(2)+1e-12).sqrt();total+=m;if u.h1_labels[s*H1+c%H1]!=u.h2_labels[s*H2+r]{off+=m}}}if total==0.0{continue}let q=off/total;let upper=(q-f64::from(u.q_targets[s])).max(0.0);let floor=(f64::from(u.q_floors[s])-q).max(0.0);let d=f64::from(u.confidences[s])*(2.0*upper-2.0*f64::from(u.beta_floor)*floor);for r in 0..H2{for c in 0..H1*2{let w=f64::from(weights[base+r*H1*2+c]);let m=(w*w+1e-12).sqrt();let is_off=u.h1_labels[s*H1+c%H1]!=u.h2_labels[s*H2+r];let g=d*(if is_off{total-off}else{-off})/(stacks as f64*total*total)*(w/m);ss+=g*g}}}ss.sqrt()}

#[cfg(test)] mod tests {use super::*;
#[test]fn block_diagonal(){let mut a=[[0.0;H1];H2];for r in 0..H2{for h in 0..H1{if (r<32)==(h<8){a[r][h]=1.0}}}assert_eq!(exact(&a).q,0.0)}
#[test]fn mapping_gradient_is_finite(){let cfg=Config{mode:Mode::Adaptive,seed:1,total_steps:100,refresh_steps:10,warmup_fraction:0.0,freeze_fraction:0.7,relax_fraction:0.9,patience_refreshes:1,n_null:2,gain_min:-1.0,gain_full:0.1,g_target:0.18,g_max:0.6,beta_floor:4.0,target_grad_ratio:0.03,lambda_min:1e-8,lambda_max:1e4,lambda_ema:0.9};let mut c=Controller::new(cfg,1).unwrap();let w=vec![0.1;H2*H1*2];let u=c.refresh(10,&w,&vec![0.01;w.len()]).unwrap();assert!(u.lambda_struct.is_finite());assert_eq!(u.h1_labels.len(),H1);assert_eq!(u.h2_labels.len(),H2);}
#[test]fn state_roundtrip_preserves_next_refresh(){let cfg=Config{mode:Mode::Adaptive,seed:7,total_steps:100,refresh_steps:10,warmup_fraction:0.0,freeze_fraction:0.7,relax_fraction:0.9,patience_refreshes:1,n_null:2,gain_min:-1.0,gain_full:0.1,g_target:0.18,g_max:0.6,beta_floor:4.0,target_grad_ratio:0.03,lambda_min:1e-8,lambda_max:1e4,lambda_ema:0.9};let w:Vec<f32>=(0..H2*H1*2).map(|i|0.01+(i%31)as f32*0.001).collect();let g=vec![0.02;w.len()];let mut uninterrupted=Controller::new(cfg.clone(),1).unwrap();uninterrupted.refresh(10,&w,&g).unwrap();let path=std::env::temp_dir().join(format!("bulletou-softmod-state-{}.txt",std::process::id()));uninterrupted.save_state(&path).unwrap();let mut resumed=Controller::load_state(cfg,1,&path).unwrap();let expected=uninterrupted.refresh(20,&w,&g).unwrap();let actual=resumed.refresh(20,&w,&g).unwrap();assert_eq!(actual,expected);std::fs::remove_file(path).unwrap();}
}
