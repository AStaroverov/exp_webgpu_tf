//! Actor-critic network: a shared MLP trunk with two heads — a policy head
//! (action logits) and a value head (state value estimate). Generic over the
//! burn `Backend` so the same definition trains under `Autodiff<Wgpu>` and runs
//! inference under the plain inner backend.

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{activation, backend::Backend, Tensor};

#[derive(Module, Debug)]
pub struct ActorCritic<B: Backend> {
    trunk: Linear<B>,
    policy: Linear<B>,
    value: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ActorCriticConfig {
    pub obs: usize,
    pub hidden: usize,
    pub actions: usize,
}

impl ActorCriticConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ActorCritic<B> {
        ActorCritic {
            trunk: LinearConfig::new(self.obs, self.hidden).init(device),
            policy: LinearConfig::new(self.hidden, self.actions).init(device),
            value: LinearConfig::new(self.hidden, 1).init(device),
        }
    }
}

impl<B: Backend> ActorCritic<B> {
    /// `x`: `[batch, obs]` → `(logits [batch, actions], value [batch, 1])`.
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = activation::relu(self.trunk.forward(x));
        let logits = self.policy.forward(h.clone());
        let value = self.value.forward(h);
        (logits, value)
    }
}
