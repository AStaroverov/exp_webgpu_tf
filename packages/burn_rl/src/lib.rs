//! PPO in Burn, in the browser.
//!
//! A self-contained test of the full RL loop running on the WebGPU backend
//! compiled to wasm: rollout collection, GAE, clipped-surrogate PPO updates with
//! autodiff + Adam, and inference. Drives a tiny grid-world environment.
//!
//! Everything async-reads tensors back from the GPU (`into_data_async`) because
//! the synchronous readbacks deadlock under wasm.

mod env;
mod model;
mod ppo;
mod rng;

use burn::backend::wgpu::{init_setup_async, Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{activation, Int, Tensor};
use wasm_bindgen::prelude::*;

use env::{GridWorld, N_ACTIONS};
use model::{ActorCritic, ActorCriticConfig};
use ppo::PpoConfig;
use rng::Rng;

/// Inference backend (plain WebGPU) and training backend (autodiff on top).
type InferBackend = Wgpu;
type TrainBackend = Autodiff<Wgpu>;
type Opt = OptimizerAdaptor<Adam, ActorCritic<TrainBackend>, TrainBackend>;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Stats returned to JS after each training iteration.
#[wasm_bindgen]
pub struct IterStats {
    pub avg_return: f32,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub episodes: u32,
    pub steps: u32,
}

#[wasm_bindgen]
pub struct Trainer {
    device: WgpuDevice,
    model: ActorCritic<TrainBackend>,
    optim: Opt,
    env: GridWorld,
    rng: Rng,
    cfg: PpoConfig,
    obs_dim: usize,
}

#[wasm_bindgen]
impl Trainer {
    /// Async constructor: initialises the WebGPU device, then builds the network
    /// and optimiser. In JS: `const t = await Trainer.create(5, 64, 42);`
    pub async fn create(grid_size: usize, hidden: usize, seed: u32) -> Trainer {
        let device = WgpuDevice::default();
        init_setup_async::<burn::backend::wgpu::graphics::AutoGraphicsApi>(
            &device,
            Default::default(),
        )
        .await;

        let env = GridWorld::new(grid_size);
        let obs_dim = env.obs_dim();
        let model = ActorCriticConfig {
            obs: obs_dim,
            hidden,
            actions: N_ACTIONS,
        }
        .init::<TrainBackend>(&device);
        let optim = AdamConfig::new().init();

        Trainer {
            device,
            model,
            optim,
            env,
            rng: Rng::new(seed as u64),
            cfg: PpoConfig::default(),
            obs_dim,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn grid_size(&self) -> usize {
        self.env.size()
    }

    #[wasm_bindgen(getter)]
    pub fn n_actions(&self) -> usize {
        N_ACTIONS
    }

    /// Collect one rollout and run `epochs` PPO updates on it.
    pub async fn train_iteration(&mut self) -> IterStats {
        let n = self.cfg.steps;

        // ---- Rollout (acting uses the non-autodiff "valid" copy of the net) ----
        let infer = self.model.valid();

        let mut obs_buf: Vec<f32> = Vec::with_capacity(n * self.obs_dim);
        let mut actions: Vec<i32> = Vec::with_capacity(n);
        let mut old_logp: Vec<f32> = Vec::with_capacity(n);
        let mut values: Vec<f32> = Vec::with_capacity(n);
        let mut rewards: Vec<f32> = Vec::with_capacity(n);
        let mut dones: Vec<bool> = Vec::with_capacity(n);

        let mut ep_return = 0.0f32;
        let mut ep_returns: Vec<f32> = Vec::new();

        let mut obs = self.env.reset(&mut self.rng);
        for _ in 0..n {
            let (probs, value) = act(&infer, &self.device, &obs).await;
            let action = self.rng.sample_categorical(&probs);
            let logp = probs[action].max(1e-8).ln();

            let step = self.env.step(action);

            obs_buf.extend_from_slice(&obs);
            actions.push(action as i32);
            old_logp.push(logp);
            values.push(value);
            rewards.push(step.reward);
            dones.push(step.done);
            ep_return += step.reward;

            if step.done {
                ep_returns.push(ep_return);
                ep_return = 0.0;
                obs = self.env.reset(&mut self.rng);
            } else {
                obs = self.env.obs();
            }
        }

        // Bootstrap value for the final (possibly non-terminal) state.
        let (_, last_value) = act(&infer, &self.device, &obs).await;

        // ---- Advantages / returns (CPU) ----
        let (mut adv, returns) = ppo::compute_gae(
            &rewards,
            &values,
            &dones,
            last_value,
            self.cfg.gamma,
            self.cfg.lambda,
        );
        ppo::normalize(&mut adv);

        // ---- Tensors for the update ----
        let dev = &self.device;
        let obs_t = Tensor::<TrainBackend, 1>::from_floats(obs_buf.as_slice(), dev)
            .reshape([n, self.obs_dim]);
        let act_t =
            Tensor::<TrainBackend, 1, Int>::from_ints(actions.as_slice(), dev).reshape([n, 1]);
        let old_logp_t =
            Tensor::<TrainBackend, 1>::from_floats(old_logp.as_slice(), dev).reshape([n, 1]);
        let adv_t = Tensor::<TrainBackend, 1>::from_floats(adv.as_slice(), dev).reshape([n, 1]);
        let ret_t = Tensor::<TrainBackend, 1>::from_floats(returns.as_slice(), dev).reshape([n, 1]);

        // ---- PPO epochs ----
        let mut model = self.model.clone();
        let mut last_pl = 0.0f32;
        let mut last_vl = 0.0f32;
        let mut last_ent = 0.0f32;

        for _ in 0..self.cfg.epochs {
            let (logits, value) = model.forward(obs_t.clone());
            let logp_all = activation::log_softmax(logits.clone(), 1);
            let probs = activation::softmax(logits, 1);

            let new_logp = logp_all.clone().gather(1, act_t.clone());
            let entropy = -(probs * logp_all).sum_dim(1).mean();

            let ratio = (new_logp - old_logp_t.clone()).exp();
            let surr1 = ratio.clone() * adv_t.clone();
            let surr2 = ratio.clamp(1.0 - self.cfg.clip, 1.0 + self.cfg.clip) * adv_t.clone();
            // Elementwise min(surr1, surr2): where surr1 > surr2 take surr2.
            let mask = surr1.clone().greater(surr2.clone());
            let policy_loss = surr1.mask_where(mask, surr2).mean().neg();

            let value_loss = (value - ret_t.clone()).powf_scalar(2.0).mean();

            let loss = policy_loss.clone()
                + value_loss.clone().mul_scalar(self.cfg.vf_coef)
                - entropy.clone().mul_scalar(self.cfg.ent_coef);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = self.optim.step(self.cfg.lr, model, grads);

            last_pl = read_scalar(policy_loss).await;
            last_vl = read_scalar(value_loss).await;
            last_ent = read_scalar(entropy).await;
        }
        self.model = model;

        let episodes = ep_returns.len();
        let avg_return = if episodes > 0 {
            ep_returns.iter().sum::<f32>() / episodes as f32
        } else {
            0.0
        };

        IterStats {
            avg_return,
            policy_loss: last_pl,
            value_loss: last_vl,
            entropy: last_ent,
            episodes: episodes as u32,
            steps: n as u32,
        }
    }

    /// Action probabilities for a single cell (length `n_actions`).
    pub async fn infer(&self, cell: usize) -> Vec<f32> {
        let infer = self.model.valid();
        let obs = self.env.obs_of(cell);
        let (probs, _) = act(&infer, &self.device, &obs).await;
        probs
    }

    /// Greedy (argmax) action for every cell — for visualising the learned policy.
    pub async fn greedy_actions(&self) -> Vec<u32> {
        let (logits, _) = self.forward_all_cells().await;
        let a = N_ACTIONS;
        (0..logits.len() / a)
            .map(|i| {
                let row = &logits[i * a..i * a + a];
                let mut best = 0usize;
                for j in 1..a {
                    if row[j] > row[best] {
                        best = j;
                    }
                }
                best as u32
            })
            .collect()
    }

    /// Value estimate for every cell — for visualising the value function.
    pub async fn values(&self) -> Vec<f32> {
        let (_, values) = self.forward_all_cells().await;
        values
    }

    /// Forward every cell at once. Returns (flat logits `[cells * n_actions]`,
    /// values `[cells]`).
    async fn forward_all_cells(&self) -> (Vec<f32>, Vec<f32>) {
        let infer = self.model.valid();
        let cells = self.obs_dim; // grid has obs_dim == cell count
        let mut flat = Vec::with_capacity(cells * self.obs_dim);
        for c in 0..cells {
            flat.extend_from_slice(&self.env.obs_of(c));
        }
        let x = Tensor::<InferBackend, 1>::from_floats(flat.as_slice(), &self.device)
            .reshape([cells, self.obs_dim]);
        let (logits, value) = infer.forward(x);
        let logits = read_all(logits).await;
        let value = read_all(value).await;
        (logits, value)
    }
}

/// Forward a single observation and return `(action_probs, value)`.
async fn act(
    model: &ActorCritic<InferBackend>,
    device: &WgpuDevice,
    obs: &[f32],
) -> (Vec<f32>, f32) {
    let x = Tensor::<InferBackend, 1>::from_floats(obs, device).reshape([1, obs.len()]);
    let (logits, value) = model.forward(x);
    let probs = activation::softmax(logits, 1);
    let probs = read_all(probs).await;
    let value = read_scalar_infer(value).await;
    (probs, value)
}

async fn read_all<const D: usize>(t: Tensor<InferBackend, D>) -> Vec<f32> {
    t.into_data_async().await.unwrap().iter::<f32>().collect()
}

async fn read_scalar_infer<const D: usize>(t: Tensor<InferBackend, D>) -> f32 {
    t.into_data_async()
        .await
        .unwrap()
        .iter::<f32>()
        .next()
        .unwrap_or(0.0)
}

async fn read_scalar<const D: usize>(t: Tensor<TrainBackend, D>) -> f32 {
    t.into_data_async()
        .await
        .unwrap()
        .iter::<f32>()
        .next()
        .unwrap_or(0.0)
}
