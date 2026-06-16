//! Actor-critic networks.
//!
//! - `ActorCritic` — the original tiny MLP trunk used by the GridWorld demo
//!   driver in `lib.rs`. Kept while the v4 migration lands.
//! - `ActorCriticV4` — the v4 Perceiver network: a board `[B,11,11,22]` + mask
//!   `[B,121]` projected to tokens `[B,121,D]`, a `HexRingGather` (37 cells),
//!   stacked `PerceiverBlock`s (4 policy / 2 value, separate instances), a final
//!   `RMSNorm`, then a policy head (concat of hold/move/fire → `[B,43]` logits)
//!   and a value head (flatten → dense → `[B,1]`).

pub mod hex;
pub mod norm;
pub mod perceiver;
pub mod posenc;

use burn::config::Config;
use burn::module::{Module, ModuleMapper};
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::tensor::{activation, backend::Backend, Tensor};

/// Identity mapper whose only effect is to force every parameter to materialise.
///
/// Burn parameters from `LinearConfig::init` etc. are *lazy*: the value is a
/// deferred RNG initializer, not a concrete tensor. `Param::clone` of a lazy
/// param duplicates the initializer (see burn-core `param/base.rs` —
/// "Each clone holds its own `Uninitialized` state"), so each independent clone
/// re-runs the RNG and produces DIFFERENT weights. Running `Module::map` with
/// this no-op mapper triggers `Param::consume` on every param (the default
/// `map_float`/`map_int`/`map_bool`), materialising it into a concrete,
/// initialized value. Subsequent `clone()`s then copy that value, so two clones
/// of the same model are bit-identical and the forward pass is deterministic.
struct Materialize;
impl<B: Backend> ModuleMapper<B> for Materialize {}

use hex::HexRingGather;
use norm::{RMSNorm, RMSNORM_EPS};
use perceiver::{PerceiverBlock, PerceiverBlockConfig};

use crate::constants::{
    ACTION_DIM_TOTAL, ACTION_CELL_INDEXES_LEN, BOARD_CELLS, BOARD_CHANNELS, FIRE_TARGET_COUNT,
    LOGIT_INIT_GAIN, MOVE_DIR_COUNT, PERCEIVER_POLICY_DEPTH, PERCEIVER_POLICY_DIM,
    PERCEIVER_POLICY_HEADS, PERCEIVER_VALUE_DEPTH, PERCEIVER_VALUE_DIM, PERCEIVER_VALUE_HEADS,
};

// ---------------------------------------------------------------------------
// Original MLP actor-critic (GridWorld demo).
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// v4 Perceiver actor-critic.
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ActorCriticV4<B: Backend> {
    // --- policy branch (dim 256, 8 heads, depth 4) ---
    pub policy_proj: Linear<B>,
    pub policy_gather: HexRingGather<B>,
    pub policy_perceiver: Vec<PerceiverBlock<B>>,
    pub policy_norm: RMSNorm<B>,
    pub policy_head_hold: Linear<B>,
    pub policy_head_move: Linear<B>,
    pub policy_head_fire: Linear<B>,

    // --- value branch (dim 128, 4 heads, depth 2) ---
    pub value_proj: Linear<B>,
    pub value_gather: HexRingGather<B>,
    pub value_perceiver: Vec<PerceiverBlock<B>>,
    pub value_norm: RMSNorm<B>,
    pub value_head: Linear<B>,
}

impl<B: Backend> ActorCriticV4<B> {
    /// Build the v4 network with the constants from [`crate::constants`].
    ///
    /// Geometry is identical for both branches (board → `BOARD_CELLS` tokens →
    /// projection → `HexRingGather` of `ACTION_CELL_INDEXES_LEN` action cells →
    /// `PerceiverBlock`s → `RMSNorm`); only the widths/heads/depths differ.
    ///
    /// Policy branch: `dim=256`, `heads=8`, `depth=4`. Three bias-free scorers
    /// (`hold`/`move`/`fire`, each `dim → 1`) are applied per action token and the
    /// 1+6+36 results concatenated into `[B, 43]` logits (consts.ts action layout).
    ///
    /// Value branch: `dim=128`, `heads=4`, `depth=2`. The `37 × 128` action-cell
    /// latents are flattened to `[B, 4736]` and a single dense (`4736 → 1`, with
    /// bias) produces the scalar value.
    ///
    /// The scorer kernels use Burn's `Orthogonal { gain: LOGIT_INIT_GAIN }` to
    /// match the TF.js v4 `tf.initializers.orthogonal({ gain: 0.2 })` near-zero
    /// logit init; all other weights keep Burn's Linear default.
    pub fn new(device: &B::Device) -> Self {
        // Validates length, in-range, and DISTINCTNESS of the gather table
        // (a length check alone missed the all-zeros placeholder bug).
        crate::constants::assert_action_cell_indexes();

        // tokenProj: bias-free Linear(BOARD_CHANNELS -> dim), linear activation.
        let proj = |dim: usize| {
            LinearConfig::new(BOARD_CHANNELS, dim)
                .with_bias(false)
                .init::<B>(device)
        };

        let perceiver = |dim: usize, heads: usize, depth: usize| {
            let cfg = PerceiverBlockConfig::new(dim, heads);
            (0..depth).map(|_| cfg.init::<B>(device)).collect::<Vec<_>>()
        };

        // Near-zero scorer init: orthogonal with the v4 logit gain, no bias.
        let scorer = |dim: usize| {
            LinearConfig::new(dim, 1)
                .with_bias(false)
                .with_initializer(Initializer::Orthogonal {
                    gain: LOGIT_INIT_GAIN,
                })
                .init::<B>(device)
        };

        Self {
            // --- policy branch -------------------------------------------------
            policy_proj: proj(PERCEIVER_POLICY_DIM),
            policy_gather: HexRingGather::new(device),
            policy_perceiver: perceiver(
                PERCEIVER_POLICY_DIM,
                PERCEIVER_POLICY_HEADS,
                PERCEIVER_POLICY_DEPTH,
            ),
            policy_norm: RMSNorm::new(PERCEIVER_POLICY_DIM, RMSNORM_EPS, device),
            policy_head_hold: scorer(PERCEIVER_POLICY_DIM),
            policy_head_move: scorer(PERCEIVER_POLICY_DIM),
            policy_head_fire: scorer(PERCEIVER_POLICY_DIM),

            // --- value branch --------------------------------------------------
            value_proj: proj(PERCEIVER_VALUE_DIM),
            value_gather: HexRingGather::new(device),
            value_perceiver: perceiver(
                PERCEIVER_VALUE_DIM,
                PERCEIVER_VALUE_HEADS,
                PERCEIVER_VALUE_DEPTH,
            ),
            value_norm: RMSNorm::new(PERCEIVER_VALUE_DIM, RMSNORM_EPS, device),
            // value head: flatten(ACTION_CELL_INDEXES_LEN * dim) -> 1, with bias.
            value_head: LinearConfig::new(ACTION_CELL_INDEXES_LEN * PERCEIVER_VALUE_DIM, 1)
                .init::<B>(device),
        }
        // Force all lazy params to materialise NOW so the model's weights are
        // concrete and stable. Without this, `clone()` re-runs each param's RNG
        // initializer independently (Burn lazy-param semantics), making two
        // clones — hence two forwards — diverge by ~1e-1. See `Materialize`.
        .map(&mut Materialize)
    }

    /// `board`: `[B, 11, 11, 22]`; `mask`: `[B, 121]`.
    /// Returns `(logits [B, 43], value [B, 1])`.
    pub fn forward(
        &self,
        board: Tensor<B, 4>,
        mask: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let batch = board.dims()[0];

        // [B, 11, 11, 22] -> [B, 121, 22]: one token per board cell.
        let tokens = board.reshape([batch, BOARD_CELLS, BOARD_CHANNELS]);

        let logits = self.policy_forward(tokens.clone(), mask.clone());
        let value = self.value_forward(tokens, mask);
        (logits, value)
    }

    /// Run a branch trunk: project tokens, gather the action cells as the
    /// perceiver latents (cross-attending the full board), stack the blocks, then
    /// the output `RMSNorm`. Returns the encoded action-cell latents `[B, 37, dim]`.
    fn encode(
        &self,
        proj: &Linear<B>,
        gather: &HexRingGather<B>,
        blocks: &[PerceiverBlock<B>],
        norm: &RMSNorm<B>,
        tokens: Tensor<B, 3>,
        mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        // Project every board cell to the branch width: [B, 121, dim] (K/V tokens).
        let projected = proj.forward(tokens);

        // Latent seed = the 37 action cells gathered from the projected board.
        let mut latents = gather.forward(projected.clone());

        // Perceiver blocks: latents read the board (content-masked cross-attn),
        // then self-attend. `kvMask` is the board content mask; the query-side
        // mask is all-ones in v4 and therefore omitted (see perceiver.rs).
        for block in blocks {
            latents = block.forward(latents, projected.clone(), mask.clone());
        }

        // Output RMSNorm over the latent features.
        norm.forward(latents)
    }

    /// Policy trunk + the hold/move/fire scorers concatenated to `[B, 43]`.
    fn policy_forward(&self, tokens: Tensor<B, 3>, mask: Tensor<B, 2>) -> Tensor<B, 2> {
        let encoded = self.encode(
            &self.policy_proj,
            &self.policy_gather,
            &self.policy_perceiver,
            &self.policy_norm,
            tokens,
            mask,
        ); // [B, 37, dim]
        let batch = encoded.dims()[0];

        // hold: latent 0. -> [B, 1, dim] -> scorer -> [B, 1, 1] -> [B, 1].
        let hold_tok = encoded.clone().slice([0..batch, 0..1]);
        let hold = self.policy_head_hold.forward(hold_tok).reshape([batch, 1]);

        // move: latents 1..1+6 (ring-1 cells). Shared scorer applied per token.
        let move_tok = encoded
            .clone()
            .slice([0..batch, MOVE_TOKEN_OFFSET..MOVE_TOKEN_OFFSET + MOVE_DIR_COUNT]);
        let move_logits = self
            .policy_head_move
            .forward(move_tok)
            .reshape([batch, MOVE_DIR_COUNT]);

        // fire: latents 1..1+36 (all ring cells). Shared scorer applied per token.
        let fire_tok = encoded.slice([
            0..batch,
            FIRE_TOKEN_OFFSET..FIRE_TOKEN_OFFSET + FIRE_TARGET_COUNT,
        ]);
        let fire_logits = self
            .policy_head_fire
            .forward(fire_tok)
            .reshape([batch, FIRE_TARGET_COUNT]);

        // Concat in consts.ts order: [hold(1), move(6), fire(36)] = [B, 43].
        let logits = Tensor::cat(vec![hold, move_logits, fire_logits], 1);
        debug_assert_eq!(logits.dims()[1], ACTION_DIM_TOTAL);
        logits
    }

    /// Value trunk: flatten the 37 latents, then a dense scalar head -> `[B, 1]`.
    fn value_forward(&self, tokens: Tensor<B, 3>, mask: Tensor<B, 2>) -> Tensor<B, 2> {
        let encoded = self.encode(
            &self.value_proj,
            &self.value_gather,
            &self.value_perceiver,
            &self.value_norm,
            tokens,
            mask,
        ); // [B, 37, dim]
        let batch = encoded.dims()[0];

        // flatten [B, 37, dim] -> [B, 37*dim], then dense -> [B, 1].
        let feature = encoded.reshape([batch, ACTION_CELL_INDEXES_LEN * PERCEIVER_VALUE_DIM]);
        self.value_head.forward(feature)
    }
}

/// First action-cell latent that carries a *move* logit: the centre cell (hold)
/// is latent 0, so the 6 ring-1 move cells start at latent 1.
const MOVE_TOKEN_OFFSET: usize = 1;
/// First action-cell latent that carries a *fire* logit. The 36 fire targets are
/// the ring cells 1..3, i.e. latents `1..37` — same start as the move slice.
const FIRE_TOKEN_OFFSET: usize = 1;
