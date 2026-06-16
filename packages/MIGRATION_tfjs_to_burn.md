I'll write the migration document directly as my output. The maps provide comprehensive ground truth, so I don't need to explore the codebase further — the task is synthesis.

# Migration: TensorFlow.js → Burn (Rust / wasm / WebGPU)

Porting the `ppo` (generic PPO+Retrace algorithm) and `ppo_unknown` (game-specific actor/critic harness) TypeScript/tfjs packages onto the existing `packages/burn_rl` Burn 0.21 foundation, while preserving the PPO+Retrace spec and the ECS-game integration.

---

## 1. Executive summary

**What is moving.** The neural-compute and model layer of two packages:

- `packages/ppo` (~5000 LOC) — the generic algorithm: separated policy/value learners, V-trace/Retrace(λ) targets, replay sampling, custom Keras layers, AdamW, KL-adaptive LR, IndexedDB persistence, RxJS channel orchestration.
- `packages/ppo_unknown` (~4700 LOC) — the game harness: the live **v4** Perceiver actor-critic over an 11×11×22 egocentric hex board, multi-worker actor/learner runtime, ECS observation snapshotting, action masking, reward shaping, curriculum learning, metrics persistence.

The compute/model layer (forward, loss, optimizer, autodiff, sampling, serialization) moves into Rust/Burn behind a wasm boundary. The **ECS game itself (bitecs + rapier), observation building, reward shaping, and curriculum logic stay in TypeScript** — they are coupled to the live game world and are pure CPU arithmetic, not tensor work.

**Why Burn.** The `burn_rl` POC already proves the hard part works in the target runtime: end-to-end PPO (rollout → GAE → clipped-surrogate → Adam → `backward()` → inference) under `burn-wgpu` compiled to wasm, with async tensor readback (`into_data_async`) to avoid wasm deadlock. Burn gives us a typed, RAII-managed tensor graph (no `tf.tidy`/`dispose` bookkeeping), off-the-shelf `AdamW`, attention, and `record`-based serialization, and removes the tfjs WebGPU `multinomial` seed-zero bug we currently work around.

**Headline risks (residual, after OSS reuse):**

1. **Custom hex layers** (`HexRingGatherLayer`, `HexNeighborGatherLayer`) and the **Perceiver composition** — no Burn drop-in; hand-build as `Module`s. burn_rl has not yet proven attention+mask training in wasm.
2. **GPU action masking + categorical sampling** — additive `-1e9` masked softmax + a stable categorical sampler; not in burn_rl.
3. **Cross-worker model/weight sharing in wasm** — no `BroadcastChannel` from Rust; must bridge via JS `postMessage` + serde.
4. **Browser persistence** — Burn has no IndexedDB backend; must bind via `wasm-bindgen`/`web-sys` or relocate to JS.
5. **Retrace(λ) / KL-adaptive LR / gradient global-norm clipping** — no OSS equivalents, but each is a small, well-specified Rust port.

**Effort shape.** Two subsystems are `very-hard` (PPO core orchestration; reward/curriculum/metrics persistence), four are `hard`, two are `moderate`, one is `trivial`. The realistic critical path is: **port v4 + custom layers → single-thread training parity on one scenario → masking/sampling → workers → persistence/curriculum → cutover.** Dropping dead code (below) removes ~4 network versions and ~15 unused layers from scope before we start.

---

## 2. Current architecture (tfjs)

```
┌─────────────────────────── Main tab (entry/index.ts) ───────────────────────────┐
│  spawns workers, VisTestEpisodeManager (debug render of live policy)             │
└──────────────────────────────────────────────────────────────────────────────────┘
        │ BroadcastChannel (RxJS): agentSample / episodeSample / curriculumState / metrics
        ▼
┌──────────── 4× ActorWorker ────────────┐        ┌──────── 2× LearnerWorker ───────┐
│ entry/ActorWorker.ts                    │        │ LearnerPolicyWorker.ts          │
│  EpisodeManager episode loop            │        │ LearnerValueWorker.ts           │
│  UnknownAgent.decide (policy infer)     │        │ createLearnerManager:           │
│  FrozenAgent.decide (frozen opponent)   │  rollout│   batch buffering (bufferWhile) │
│  createPolicyDriverSystem (sync ticks)  │ ──────▶ │   computeRetrace (V-trace)      │
│  ECS game (bitecs + rapier, headless)   │ samples │   createPolicyLearnerAgent      │
│  snapshotUnknownBoard → InputArrays     │        │     PPO surrogate+entropy+KL     │
│  computeActionMask → batchActAsync      │        │   createValueLearnerAgent        │
└─────────────────────────────────────────┘        │     clipped MSE                  │
        │ loads policy from IndexedDB                │   AdamW + global-norm clip       │
        ▼                                            └──────────────────────────────────┘
   tf.io indexeddb:// (latest + ≤20 historical snapshots, userDefinedMetadata)
```

**Data flow per decision point:** ECS reaches a decision tick → `snapshotUnknownBoard` fills an egocentric `[11,11,22]` board + `[121]` content mask into `InputArrays` → `computeActionMask` builds the additive mask (Hold never masked; off-board moves/fires masked) → `batchActAsync` runs the network forward, applies the mask before softmax, samples a categorical action → `applyActionToGame` decodes to Hold/MoveStep/Fire. Steps accumulate in `AgentMemory` (`addFirstPart`/`updateSecondPart`); at episode end the trajectory + `logProbs` + rewards (from `ScoreTracker` + `calculateReward`) flow to learners. Learners buffer batches, compute Retrace targets, run K epochs of mini-batch PPO (`ReplayBuffer` shuffles indices), monitor KL to adapt LR (`getDynamicLearningRate`), checkpoint to IndexedDB, and broadcast metrics. The frozen opponent is a random historical snapshot re-rolled per episode for stable self-play.

**The networks:** the only live one is **v4** (`packages/ppo_unknown/src/models/Networks/v4.ts`): a Perceiver over 37 action cells (self + fire rings 1–3), policy depth 4 / 8 heads, value depth 2 / 4 heads, single flat 43-logit head (1 Hold + 6 moves + 36 fire targets). Pre-LN residual stream, RMSNorm, SiLU-gated FFN, Grid2D positional encoding, `HexRingGatherLayer` for fixed action-cell selection.

---

## 3. Target architecture (Burn / Rust / wasm / wgpu)

**What burn_rl already proves** (`packages/burn_rl/src/`):

| File | Proven capability |
|---|---|
| `lib.rs` | `Trainer` struct, `train_iteration` (rollout + PPO epochs), inference, `wasm-bindgen` bindings, **async readback** to avoid wasm deadlock |
| `model.rs` | `ActorCritic` (shared MLP trunk + policy/value heads), `Module` derive, `Config` init |
| `ppo.rs` | `PpoConfig`, `compute_gae` (CPU scalar), advantage `normalize` |
| `env.rs` | toy GridWorld (will be *replaced* — the real env stays in TS) |
| `rng.rs` | seeded `xorshift64*`, categorical sampling, no `getrandom` on hot path |
| `Cargo.toml` | burn 0.21 + `burn-wgpu` + `Autodiff`, `init_setup_async` |

It does **not** yet prove: multi-head/flat masked logits, masked softmax + categorical sampling, attention/transformer/Perceiver blocks, the custom hex-gather layers, `burn::record` save/load round-trip, AdamW (uses plain Adam), gradient clipping, or any cross-worker / persistence story.

**Proposed end-state crate layout** (extend the existing crate; one wasm module, instantiated per worker):

```
packages/burn_rl/
  src/
    lib.rs              # wasm-bindgen surface: PolicyTrainer / ValueTrainer / Inferer
    model/
      mod.rs            # ActorCritic v4 (policy & value, param dims/depth)
      perceiver.rs      # cross-attn + self-attn block (composes burn::nn::attention)
      hex.rs            # HexRingGather, HexNeighborGather as Modules (tensor ops)
      norm.rs           # RMSNorm (hand-built; not in burn::nn 0.21)
      posenc.rs         # Grid2D sin/cos positional encoding (static tensor build)
    ppo/
      mod.rs            # PpoConfig
      gae.rs            # (existing) GAE
      retrace.rs        # GAP: V-trace/Retrace(λ) reverse scan
      loss.rs           # clipped surrogate, clipped value MSE, masked entropy, KL
      sample.rs         # masked softmax + stable categorical sampler
      clip.rs           # global-norm gradient clipping
      lr.rs             # KL-adaptive LR multiplier
    optim.rs            # AdamW config + decoupled decay exemption (bias/norm)
    record.rs           # save/load + metadata (expIteration, curriculumState)
    rng.rs              # (existing)
  index.ts              # wasm glue + IndexedDB FFI (web-sys/idb on JS side)
```

**How the TS game talks to Rust.** The ECS game, observation building, masking inputs, reward shaping and curriculum **stay in TS**. Each tick that reaches a decision point, TS hands Rust a flat `Float32Array` board batch `[B,11,11,22]`, a `[B,121]` content mask, and a `[B,43]` additive action mask; Rust returns sampled actions + log-probs + value estimates (read back async). At episode end TS sends the serialized trajectory to the learner worker; Rust trains and emits new weights + metrics. This keeps bitecs/rapier/honeycomb-grid (all JS) authoritative and avoids reimplementing hex geometry and the game in Rust.

**Worker model.** Keep the current shape: 4 actor wasm instances + 2 learner wasm instances (or a fused learner — see Open Questions). Each worker instantiates its own Burn `WgpuDevice` (`init_setup_async`). Cross-worker traffic (samples, weights, curriculum, metrics) crosses via JS `postMessage` + serde (bincode for weights, JSON for metrics/curriculum) — Rust cannot use `BroadcastChannel` directly.

---

## 4. Subsystem-by-subsystem migration plan

Legend: **OSS** = off-the-shelf Burn/ecosystem crate; **GAP** = hand-build (justified); **TS-STAYS** = remains TypeScript, does not port.

### 4.1 PPO core training orchestration (`packages/ppo/src/core`, `/learner`, `/memory`)

| tfjs feature | Burn 0.21 equivalent | Kind |
|---|---|---|
| `tf.LayersModel.apply(states,{training})` | `Module::forward` + train/eval mode | OSS `burn::nn` |
| `tf.variableGrads` + `tf.tidy`/`dispose` | `tensor.backward()` + RAII (no dispose) | OSS `burn::autodiff` |
| `tf.Optimizer.applyGradients` (Adam) | `burn::optim::AdamW` via `OptimizerAdaptor` + `GradientsParams` | OSS |
| custom global-norm gradient clip in `optimize()` | **clip.rs**: Σgrad² → sqrt → `coef=min(1,clipNorm/norm)` → scale | GAP |
| categorical sample (`softmax`+`multinomial`) | **sample.rs**: `activation::softmax` + CDF sampler over `rng.rs` | GAP |
| `computeRetrace` / V-trace (`train.ts`) | **retrace.rs** reverse scan, CPU scalar | GAP |
| PPO clipped surrogate / clipped value MSE | **loss.rs** (`exp`,`clamp`,`min`,`mean`,`neg`; `MseLoss`) | OSS ops + GAP glue |
| KL (Schulman `r-1-log r`) + `getDynamicLearningRate` | **lr.rs**: ring buffer median KL → ×0.95/×1.05 on `AdamW` lr | GAP |
| `ReplayBuffer` shuffled mini-batches | index `Vec` + `rng.rs` permutation | GAP (trivial) |
| `Memory.ts` `AgentMemory` two-phase trajectory | Rust struct of `Vec`/`Tensor1D`; serde | GAP (trivial) |
| RxJS channels (`bufferWhile`,`scan`,`mergeMap`) | TS-side orchestration + `postMessage`; Rust stays request/response | **TS-STAYS** mostly |
| `metricsChannels`, `analyzeVTrace` | metrics computed in Rust, emitted as JSON to TS chart | GAP (stats) |
| IndexedDB save/load (`storage.ts`,`Transfer.ts`) | `burn::record` bincode + JS FFI to IndexedDB | OSS + GAP |
| `modelsCopy` Adam-state clone | `burn::record` snapshot/restore of optimizer state | OSS |

Effort: **very-hard** (core) / **hard** (learner agents).

### 4.2 Model system — networks, layers, optimizer (`packages/ppo/src/models`, `ppo_unknown/.../v4.ts`)

| tfjs feature | Burn 0.21 equivalent | Kind |
|---|---|---|
| `MultiHeadAttentionLayer` | `burn::nn::attention::MultiHeadAttention` (+ custom wrapper for fused self-attn) | OSS |
| `PerceiverLayer` / `CrossTransformer` (functions) | compose attention + FFN as a `Module` (**perceiver.rs**) | OSS-composed |
| `RMSNormLayer` | **norm.rs** hand-built RMSNorm (no `RMSNorm` in burn::nn 0.21) | GAP |
| `createDenseLayer` | `burn::nn::Linear` + activation | OSS |
| `Grid2DPositionalEncodingLayer` (scale 0.1) | **posenc.rs** static sin/cos tensor | GAP |
| `SlotEmbeddingLayer` (latent seed) | `burn::nn::Embedding` | OSS |
| `HexRingGatherLayer` (fixed `ACTION_CELL_INDEXES`) | **hex.rs** `tensor.gather`/index-select with precomputed indices | GAP |
| `HexNeighborGatherLayer` (matmul × neighbor matrix) | **hex.rs** `matmul` with precomputed neighbor matrix | GAP |
| `AdamW` (decoupled decay, skip bias/norm/σ) | `burn::optim::AdamW` + per-param exemption filter (**optim.rs**) | OSS + GAP filter |
| `PatchedAdamOptimizer` (name-keyed moments, dynamic layers) | not needed — v4 is static; standard AdamW state | DROP need |
| init `orthogonal`/`glorotUniform`, `eps=1e-5` | `burn::nn::Initializer`; set AdamW `epsilon=1e-5` | OSS |
| `tf.model` functional + `registerClass` | `Module` derive + `burn::record` (no registry) | OSS |

Effort: **hard**.

### 4.3 Memory & noise (`packages/ppo/src/memory`, `/noise`)

| tfjs feature | Burn equivalent | Kind |
|---|---|---|
| `AgentMemory` trajectory | Rust struct + serde | GAP (trivial) |
| `ReplayBuffer` uniform shuffle | index `Vec` + `rng.rs` | GAP (trivial) |

Effort: **trivial**. (All exploration noise — the standalone classes **and** `NoisyDenseLayer` with its `ColoredNoiseState` — is dead; see §4b. Nothing noise-related migrates.)

### 4.4 ppo_unknown ECS driving & game integration (`packages/ppo_unknown/src/env`, `/state`)

| tfjs feature | Disposition | Kind |
|---|---|---|
| `createUnknownScenario`, `createPolicyDriverSystem`, `applyActionToGame` | stays TS (ECS-coupled) | **TS-STAYS** |
| `computeActionMask` (Hold=0, passability, gun/friendly) | stays TS; emits additive mask array to Rust | **TS-STAYS** |
| `snapshotUnknownBoard`, `board.ts`, `InputArrays`, `markBulletThreat`, `hexNeighbors` | stays TS (bitecs/honeycomb) — fills `Float32Array` | **TS-STAYS** |
| `InputTensors.ts` (`tf.tensor4d`/`tensor2d`) | **drops** — Rust builds tensors from the `Float32Array` | DROP→Rust |
| `UnknownAgent.decide` / `FrozenAgent.decide` inference | TS orchestrates; Rust does forward+sample | split |
| `batchActAsync` | Rust inference + masked sample + async readback | GAP |

Effort: **hard** (mainly the inference boundary + custom layers it pulls from ppo).

### 4.5 Reward, curriculum, metrics (`packages/ppo_unknown/src/reward`, `/curriculum`, `/ui/MetricsBrowser`)

| tfjs feature | Disposition | Kind |
|---|---|---|
| `calculateReward`, `getShapingWeight`, `ScoreTracker` (hex dist via `MapDI.grid`, bitecs) | **stays TS** — pure arithmetic, ECS-coupled; pass scalar rewards to Rust | **TS-STAYS** |
| `createScenarioByCurriculumState` (threshold anneal, unlock, softmax sampling) | **stays TS** — pure logic | **TS-STAYS** |
| `curriculumMeta` (tf `userDefinedMetadata`) | persist alongside checkpoint: serde struct in `record.rs` **or** keep in JS IndexedDB | GAP/decision |
| `MetricsBrowser` Dexie + Chart.js | **stays TS** (browser-side); Rust just emits metric JSON | **TS-STAYS** |
| `consts.ts` action dims (43, 6, 36) | mirror as Rust consts | GAP (trivial) |

Effort: **very-hard** label is dominated by the persistence-bridge + masking integration, not the arithmetic (which stays TS).

---

## 4b. Dead code to DROP (do not migrate)

Excluded from all effort/plan estimates. **Confirm before deleting** with a repo-wide `grep` for the symbol/import (the maps already report zero live imports, but verify against the actual HEAD).

| Item | Why dead | Confirm |
|---|---|---|
| `ppo_unknown/src/models/Networks/v0.ts` | baseline flatten-MLP; never imported | grep `Networks/v0` |
| `ppo_unknown/src/models/Networks/v1.ts` | early transformer, superseded by v4 | grep `Networks/v1` |
| `ppo_unknown/src/models/Networks/v2.ts` | 7-cell perceiver, superseded by v4 | grep `Networks/v2` |
| `ppo_unknown/src/models/Networks/v3.ts` | duplicate of v2 / 7-latent, superseded by v4 | grep `Networks/v3` |
| `ppo/src/memory/PrioritizedReplayBuffer.ts` | extends `ReplayBuffer`, never instantiated | grep `PrioritizedReplayBuffer` |
| `ppo/src/models/Restore.ts` (`restoreModels`,`upsertModels`) | never called; `Transfer.ts` is the live path | grep `restoreModels` |
| `ppo/src/utils/modelsCopy.ts` (`setModelState`) | exported, 0 usages | grep `setModelState` |
| `ppo/src/utils/logProb.ts` (`computeLogProb`, `computeLogProbTanh`) | Gaussian/tanh log-prob; discrete pipeline uses inline categorical | grep usages |
| `ppo/src/models/Layers/MoELayer.ts` | registered, never instantiated | grep `MoELayer` |
| `ppo/src/models/Layers/SwinAttentionLayer.ts` | never used | grep `SwinAttention` |
| `ppo/src/models/Layers/RoleEmbeddingLayer.ts` / `RoleEncodingLayer.ts` | superseded by SlotEmbedding (itself only in dead v2/v3) | grep |
| `ppo/src/models/Layers/LogStdLayer.ts` | Gaussian log-std; policy is categorical | grep `LogStd` |
| `ppo/src/models/Layers/VariableLayer.ts` | never instantiated | grep `VariableLayer` |
| `ppo/src/models/Layers/StopGradientLayer.ts` | never wired into a network | grep `StopGradient` |
| `ppo/src/models/Layers/CloneLayer.ts` | unused | grep `CloneLayer` |
| `ppo/src/models/Layers/MaskSquashLayer.ts`, `MaskPoolLayer.ts` | never instantiated | grep |
| `ppo/src/models/Layers/FixedPositionalEncodingLayer.ts` | v4 uses Grid2D / HexRingGather, not this | grep |
| `ppo/src/models/Layers/SlotEmbeddingLayer.ts` | only in dead v2/v3 | grep |
| `ppo/src/noise/{ColoredNoise,ColoredNoiseApprox,DirichletNoise,NoiseMatrix}.ts` | never imported | grep each |
| `ppo/src/models/Layers/NoisyDenseLayer.ts` (incl. its embedded `ColoredNoiseState`) | v4 uses plain Dense, never NoisyDense; not wired into any live network | grep `NoisyDense` |
| `ppo_unknown/.../HexNeighborGatherLayer.ts` | **conflicting reports** — one map says live, another says only `HexRingGatherLayer` is imported by v4. **Verify against v4.ts before porting.** | grep in `v4.ts` |
| `Grid2DPositionalEncodingLayer.ts`, `noiseGate.ts` | live *only* if v4 uses them — confirm against v4; drop if not | grep in `v4.ts` |

> Net: porting only **v4** lets us drop 4 network versions and ~15 layer files. The two "verify" rows (`HexNeighborGatherLayer`, `Grid2DPositionalEncoding`) are the only ambiguous ones — settle them by reading `v4.ts` before writing `hex.rs`/`posenc.rs`.

---

## 4c. OSS Burn-ecosystem reuse map

Default posture: **reuse**. Hand-build only the residual.

| Need | OSS crate / module (Burn 0.21) | Replaces |
|---|---|---|
| Dense layers | `burn::nn::Linear` / `LinearConfig` | `tf.layers.dense`, `createDenseLayer` |
| Multi-head attention | `burn::nn::attention::MultiHeadAttention` | `MultiHeadAttentionLayer` |
| Embedding | `burn::nn::Embedding` | `SlotEmbeddingLayer` |
| LayerNorm | `burn::nn::LayerNorm` | (RMSNorm — see GAP) |
| Activations | `burn::tensor::activation::{relu,softmax,log_softmax,silu}` | `tf.softmax`, gating |
| Optimizer | `burn::optim::AdamW` + `OptimizerAdaptor` + `GradientsParams` | custom `AdamW.ts` |
| Value loss | `burn::nn::loss::MseLoss` (or 2 tensor ops) | `meanSquaredError` |
| Autodiff | `burn::backend::Autodiff`, `tensor.backward()` | `tf.variableGrads` |
| Backend | `burn::backend::wgpu::{Wgpu,WgpuDevice,init_setup_async}`, `AutoGraphicsApi` | tfjs-backend-webgpu |
| Async readback | `burn::tensor::into_data_async()` | `dataSync`/`data` + `onSubmittedWorkDone` |
| Serialization | `burn::record::Recorder` (bincode/safetensors) + serde | `tf.io` IndexedDB save/load |
| Module/config | `burn::module::Module`, `burn::config::Config` derive | `tf.model`, `registerClass` |
| Init | `burn::nn::Initializer` (`Orthogonal`,`GlorotUniform`,`Zeros`) | `tf.initializers.*` |
| RNG | existing `rng.rs` xorshift64* | tfjs `multinomial` (buggy) |
| Worker IPC | JS `postMessage` + serde (no Rust crate) | `BroadcastChannel`/RxJS |
| (optional) weight import | `burn-import` / `safetensors` | one-time tfjs→Burn weight conversion |

**Residual that must be hand-built** (justified — no OSS exists):

| Hand-built | Why no OSS |
|---|---|
| RMSNorm (`norm.rs`) | `burn::nn` 0.21 ships LayerNorm, not RMSNorm |
| Hex gather layers (`hex.rs`) | game-specific (fixed indices / neighbor matrix) |
| Grid2D positional encoding (`posenc.rs`) | game-specific sin/cos with scale 0.1 |
| Perceiver block (`perceiver.rs`) | composition only — no single Burn module |
| Retrace(λ)/V-trace (`retrace.rs`) | no off-policy correction in Burn core |
| PPO surrogate + clipped value loss + masked entropy (`loss.rs`) | RL-specific; 2–3 tensor ops each |
| Masked softmax + categorical sampler (`sample.rs`) | no `masked_softmax`/`multinomial` in public API |
| Global-norm gradient clip (`clip.rs`) | `AdamW` has no integrated clip_norm |
| KL-adaptive LR (`lr.rs`) | application-specific schedule |
| IndexedDB FFI | Burn has no browser persistence backend |

---

## 5. Hard problems / gaps (residual, after OSS reuse)

Everything covered by §4c OSS is excluded. These are the items with **no turnkey Burn/OSS equivalent** that `burn_rl` does **not** yet prove.

1. **Custom hex layers + Perceiver composition** — *difficulty: high.* `HexRingGatherLayer` is `tensor.gather` over a precomputed `Uint32` of `ACTION_CELL_INDEXES` (37 fixed egocentric cells); `HexNeighborGatherLayer` is two `matmul`s against a precomputed neighbor matrix. Build both as `Module`s holding the index/matrix as a non-learnable buffer (constant tensor on device). The Perceiver is cross-attn(latents←tokens) → self-attn(latents) with pre-LN + input/output norm; compose from `MultiHeadAttention` + `Linear` + `norm.rs`. *Approach:* implement `hex.rs`/`perceiver.rs` first and unit-test forward shape `[B,43]` against a tfjs reference dump before any training. burn_rl has not validated attention on wasm/wgpu — de-risk in P0.

2. **GPU action masking + categorical sampling** — *difficulty: high.* tfjs adds `-1e9` to forbidden logits before softmax, then `multinomial`. In Burn: `logits.mask_where(mask_bool, neg_inf_like)` (or additive `+mask` tensor) → `activation::softmax` → CDF sample via `rng.rs`. **Hold is never masked** (guarantees ≥1 valid action, avoids all-`-1e9` NaN). *Risk:* verify softmax over a row that is mostly `-1e9` is finite; clamp if needed. Log-prob and entropy must use the **same masked** logits.

3. **Retrace(λ) / V-trace targets** — *difficulty: moderate.* Pure CPU reverse scan (the `burn_rl` `compute_gae` already proves the pattern). Port `computeRetrace` from `packages/ppo/src/core/train.ts`: clamp log-ratio to **[-20, 20]** before `exp` (overflow guard — critical), compute clipped ρ̂, accumulate trace with λ, produce TD targets + advantages. Targets are **detached** (built outside the autodiff tape — use the inference backend / plain `f32`), preventing gradient flow.

4. **Global-norm gradient clipping** — *difficulty: low.* `AdamW` has no clip_norm. After `backward()`, before `optimizer.step`: sum of squares over all `GradientsParams`, `sqrt`, `coef = min(1, clipNorm/norm)`, scale all grads (`clipNorm=5`).

5. **KL-adaptive LR** — *difficulty: low.* Ring buffer (≈25) of per-epoch KL median (Schulman `r-1-log r`); multiply `AdamW` lr by 0.95 (KL high) / 1.05 (KL low) within bounds. AdamW lr is a settable field.

6. **AdamW decay-exemption filter (bias / norm / σ)** — *difficulty: low.* **AdamW itself — including decoupled weight decay — is OSS (`burn::optim::AdamW` via `AdamWConfig`); we do not reimplement it.** Burn applies one global `weight_decay` to all params, whereas tfjs `AdamW.ts` *skips* decay for bias/norm params by name match. The only residual is that exemption: split params into a decayed / undecayed grouping (or zero the decay on the exempt subset). Thin filter on top of Burn's optimizer, not a custom optimizer.

7. **Cross-worker model/weight & metric sharing in wasm** — *difficulty: high.* No `BroadcastChannel` from Rust. Serialize weights with `burn::record` (bincode) → transfer via `postMessage` (transferable `ArrayBuffer`) on the JS side → deserialize in the target worker. Samples, curriculum state, metrics likewise cross via JS. *Risk:* copy overhead + serialize/deserialize latency on every weight push; consider delta or throttled pushes.

8. **Browser persistence (IndexedDB) from Rust** — *difficulty: high.* Burn `record` is filesystem-oriented. Keep the **IndexedDB layer in JS** (`web-sys`/`idb` or plain JS in `index.ts`): Rust emits a bincode `Vec<u8>` checkpoint + a metadata JSON (`expIteration`, `curriculumState`); JS writes it under the existing `indexeddb://`-style keying with the ≤20-snapshot eviction and latest/historical split. Frozen-opponent re-roll picks a random historical key in JS and hands the bytes to Rust to deserialize.

9. **Frozen opponent historical roll** — *difficulty: moderate.* Was `tf.io.loadLayersModel(randomHistorical)`. Now JS picks a random snapshot key, loads bytes from IndexedDB, passes to a Rust `Inferer::load(bytes)`. No re-export needed if we standardize on the bincode checkpoint format from day one.

**Explicitly NOT gaps (OSS covers them):** AdamW itself, attention primitive, MSE loss, autodiff, embeddings, serialization format, backend init/readback. **Out of scope (dead):** PrioritizedReplay, colored/Dirichlet/matrix noise, MoE, Gaussian log-std — do not build.

---

## 6. The wasm boundary & worker model

**Split of responsibilities:**

| Concern | Lives in | Rationale |
|---|---|---|
| ECS world, bitecs systems, rapier physics | **TS** | the game; not tensor work |
| Hex geometry (`honeycomb-grid`, `hexNeighbors`, neighbor/ring tables) | **TS** | already correct; generates the index/matrix *constants* shipped once to Rust |
| Observation snapshot → `Float32Array` board + content mask (`snapshotUnknownBoard`, `markBulletThreat`) | **TS** | reads live ECS state |
| Action mask construction (`computeActionMask`) | **TS** | reads live ECS state |
| Reward shaping + score tracking (`calculateReward`, `ScoreTracker`) | **TS** | reads ECS + hex distance |
| Curriculum state machine + scenario sampling | **TS** | pure logic, coupled to episode results |
| Network forward / sample / value | **Rust/Burn** | the migration target |
| PPO/Retrace loss, AdamW, autodiff, grad clip, KL-LR | **Rust/Burn** | the migration target |
| Checkpoint serialize | **Rust** (`record` bincode); **JS** does IndexedDB I/O | Burn has no browser store |

**What crosses the boundary per decision tick (actor):**

- TS → Rust: `board: Float32Array[B*11*11*22]`, `contentMask: Float32Array[B*121]`, `actionMask: Float32Array[B*43]` (additive, Hold=0).
- Rust → TS: `actions: Int32Array[B]`, `logProbs: Float32Array[B]`, `values: Float32Array[B]` — read back via `into_data_async`.

**Per episode (actor → learner):** serialized `AgentMemory` trajectory (boards/masks/actions/logProbs/rewards/dones) over `postMessage`.

**Learner → all:** new weights (bincode `ArrayBuffer`, transferable) + metric JSON; learner → JS → IndexedDB for checkpoints.

**One-time setup:** TS computes `ACTION_CELL_INDEXES` and the hex neighbor matrix and passes them to Rust at `Trainer::create` so `hex.rs` builds its constant device buffers.

**Worker model:** unchanged topology (4 actors, 2 learners), one wasm module instantiated per worker, each with its own `WgpuDevice`. Backend is compile-time `Autodiff<Wgpu>` (no runtime backend swap — accept the loss of the wasm-vs-webgpu toggle, or feature-gate it).

---

## 7. Phased rollout plan

| Phase | Goal | Scope | Exit / verification |
|---|---|---|---|
| **P0 — Real network scaffold** | v4 architecture runs forward in Burn/wasm | `model/` (perceiver.rs, hex.rs, norm.rs, posenc.rs, Embedding), Config dims matching v4 (depths 4/2, heads 8/4, 43-logit head). Confirm `HexNeighborGather`/`Grid2D` usage by reading `v4.ts`. | Forward on a fixed board batch produces `[B,43]` logits + scalar value; **matches a tfjs reference dump within tolerance** for identical (imported) weights |
| **P1 — Single-thread training parity** | One actor + one learner, one scenario, no workers | rollout (TS env via boundary) → `retrace.rs` → `loss.rs` → `AdamW` + `clip.rs` → `backward`/step; `sample.rs` masked categorical; `ReplayBuffer` indices | Reward curve on the simplest curriculum rung tracks the tfjs baseline within noise over N iters; KL/entropy/loss in sane ranges; no NaN |
| **P2 — Masking, sampling, KL-LR hardening** | Numerical correctness of the RL math | masked softmax edge cases (all-but-Hold masked), log-ratio clamp [-20,20], detached targets, `lr.rs` KL adaptation, AdamW decay exemptions | Unit tests: masked-softmax finite; entropy/KL match tfjs on dumped batches; LR moves correctly with synthetic KL |
| **P3 — Multi-worker runtime** | 4 actors + 2 learners over wasm | `postMessage`+serde sample/weight/metric bridge; per-worker device init; frozen-opponent inference path | Throughput ≥ usable; weights propagate; training stable for an extended run |
| **P4 — Persistence + curriculum** | Checkpointing + curriculum loop closed | bincode checkpoint via `record.rs`; JS IndexedDB FFI (≤20 snapshots, latest/historical, metadata incl. `curriculumState`); frozen-opponent historical roll; curriculum state machine (stays TS) wired to learner | Restart resumes from checkpoint; historical roll loads correctly; curriculum advances rungs as before |
| **P5 — Cutover & cleanup** | Burn is the only path | delete dead code (§4b), remove tfjs deps from the training path, retire `env.rs` toy world | Full self-play run reaches a target rung; tfjs training code removed; CI green |

Each phase is independently shippable: P0–P2 run single-tab; P3+ light up the worker fleet.

---

## 8. Open questions & decisions

1. **Port only v4, or keep loaders for v0–v3?** Recommendation: **v4 only.** v0–v3 are dead; historical checkpoints are tfjs-format and incompatible regardless — start a fresh Burn checkpoint lineage (optionally one-time `burn-import` of the current best v4 weights). Decide whether to seed from the existing trained v4 or train from scratch.
2. **`HexNeighborGatherLayer` & `Grid2DPositionalEncoding` — live in v4 or not?** Maps conflict. **Read `v4.ts` before P0** and port only what v4 actually instantiates.
3. **Separate policy/value learner workers, or fuse?** tfjs splits them. Fusing into one learner halves device init and the weight-sync surface but couples the two LR schedules. Recommendation: **keep separate** to preserve the proven training dynamics; revisit if worker count is a bottleneck.
4. **Drop `PrioritizedReplayBuffer` permanently?** Yes — dead and never instantiated. Build only uniform `ReplayBuffer`.
5. **Exploration noise — decided: drop entirely.** v4 uses none. All standalone noise classes **and** `NoisyDenseLayer`/`ColoredNoiseState` are dead and will not be migrated (see §4b).
6. **Persistence home: Rust-FFI IndexedDB vs JS-side IndexedDB.** Recommendation: **Rust serializes, JS persists** — least new Rust surface, reuses the proven IndexedDB keying/eviction.
7. **Backend toggle.** Drop the runtime wasm/webgpu switch (`initTensorFlow`) and compile-time fix `Autodiff<Wgpu>`, or feature-gate? Recommendation: **fix to wgpu**; CPU only for tests.
8. **Curriculum/reward stay in TS — confirm.** They are ECS/hex-coupled pure logic; reimplementing in Rust is non-trivial and gains nothing. Recommendation: **keep in TS**, pass scalar rewards + scenario choices across the boundary.
9. **Weight-sync granularity.** Push full weights every learn step (simple, copy-heavy) vs throttled/delta. Decide in P3 against measured `postMessage` cost.

---

## 9. Effort & risk register

| Subsystem | Effort | Top risks |
|---|---|---|
| PPO core training orchestration (`ppo/src/core`,`/learner`,`/memory`) | **very-hard** | global-norm clip hand-impl; categorical sampler (tfjs multinomial bug); RxJS→postMessage state machines + backpressure; detached Retrace targets; log-ratio clamp/numerics |
| Reward / curriculum / metrics persistence (`ppo_unknown/.../reward`,`/curriculum`,`/ui`) | **very-hard** | curriculum/metadata persistence has no Burn equiv (custom JS bridge); Dexie/Chart stay TS; masking in loss; ScoreTracker stays TS (ECS-coupled) |
| Model system — networks/layers/optimizer (`ppo/src/models`, `v4.ts`) | **hard** | RMSNorm not in burn::nn; fused-attn wrapper; AdamW decay exemptions by name; record round-trip unproven; attention on wasm/wgpu unproven |
| Learner manager & policy/value agents (`ppo/src/learner`) | **hard** | Retrace math correctness; IS-weight explosion (clamp fragility); per-param vs global clip; isLossDangerous guards + reload-on-error; optimizer-state transfer |
| Multi-worker actor/learner runtime (`ppo_unknown/entry`,`/env`) | **hard** | no BroadcastChannel from Rust; per-worker device contention; IndexedDB FFI; masked-softmax NaN; KL-LR feedback timing; frozen historical roll |
| ECS game driving (`ppo_unknown/src/env`,`/state`) | **hard** | Perceiver/attention not proven in Rust; async readback timing vs same-tick drain; egocentric board shape exactness; hex constants must match |
| ppo_unknown network architectures (`v4.ts`) | **moderate** | HexNeighborGather matmul constant lifecycle; Grid2D scale 0.1 fidelity; AdamW eps 1e-5; flat-categorical mask semantics; egocentric fixed indices |
| burn_rl target (extend in place) | **hard** (build-out) | grad clip + masked sampling not yet present; custom layers absent; no-std/getrandom on wasm; record untested; per-worker device init |
| Memory & (dead) noise (`ppo/src/memory`,`/noise`) | **trivial** | only AgentMemory + uniform ReplayBuffer port; everything else dead |

**Overall:** the algorithmic port (loss, Retrace, AdamW, clip, LR) is small and well-specified; the genuine schedule risk lives in (a) the **custom hex/Perceiver layers + masked sampling on wgpu/wasm** (de-risk in P0–P2) and (b) the **wasm worker IPC + IndexedDB bridge** (P3–P4), neither of which `burn_rl` currently proves.