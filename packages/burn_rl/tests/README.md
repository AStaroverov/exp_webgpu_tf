# burn_rl native tests

Run all native tests (no browser, native WebGPU = Metal on macOS):

```sh
cargo test -- --nocapture
```

Test files:

- `smoke.rs` — the P1 "no-NaN / sane ranges" gate. Drives `V4Trainer::update`
  over a synthetic rollout and asserts every returned stat is finite and in
  range. Finiteness only — it does NOT check numerical correctness.
- `correctness.rs` — the H1/H2/H3-class correctness gates:
  1. **record save/load round-trip** — serialise a v4 model via burn's
     `BinBytesRecorder<FullPrecisionSettings>`, load it into a second
     differently-initialised model, and assert the second model then serialises
     **byte-for-byte identically** to the first (and that the two pre-load models
     differ, so the test isn't vacuous). The comparison is on the serialised
     PARAMETERS, not forward outputs — comparing the recorder's own bytes is the
     exact, forward-independent persistence assertion.
  1b. **forward determinism** — `forward_is_deterministic` asserts two forwards
     of the SAME fixed model on the WebGPU backend, same input, are **bit-exact**
     (max diff `0`). See the root-cause note below.
- `cpu_determinism.rs` — `ndarray_forward_is_deterministic`, the CPU oracle for
  the same property on the NdArray backend (no GPU, no autotune), guarding the
  lazy-`Param`-clone regression directly.

## Forward non-determinism — root cause (FIXED)

Two forwards of one fixed model used to diverge by ~1e-1 on WebGPU. The cause
was **NOT** GPU floating-point non-associativity (hypothesis A) and **NOT**
cubecl autotune (hypothesis B) — it was a model-logic bug (hypothesis C):

- **Burn `Param`s from `LinearConfig::init` (and friends) are *lazy*** — the value
  is a deferred RNG initializer closure, not a concrete tensor.
- **`Param::clone` of a lazy param duplicates the *initializer*, not a value.**
  From burn-core `module/param/base.rs`: *"Each clone holds its own
  `Uninitialized` state, so initializing one does not affect the other."* So two
  independent `model.clone()`s, on first `.val()`, each re-ran the RNG and got
  DIFFERENT weights — and the two forwards diverged.
- **Proof it was code, not the GPU:** the CPU `NdArray` backend diverged
  identically (~5e-1), and a single bare `Linear` cloned twice produced weights
  differing by ~0.55. Per-layer recomputation on the *same* instance was bit-exact
  at every depth (proj/gather/1–4 perceiver blocks all `0`), isolating the issue
  to `clone()`, not any kernel.

**Fix:** `ActorCriticV4::new` now runs `Module::map(&mut Materialize)` (a no-op
mapper that forces `Param::consume` on every parameter) before returning, so all
params are concrete/initialized and `clone()` copies the values. Both backends
now report max diff `0`.

**Training-loop implication:** the rollout's `old_logp` and the epoch-0
recomputed `new_logp` come from the same materialized weights, so at the first
epoch the importance ratio is exactly 1 — no ~1e-1 logit noise. The `V4Trainer`
builds its model via `ActorCriticV4::new`, so it inherits the fix automatically;
no separate trainer change was needed.
  2. **HexRingGather structural test** — feeds a board whose token features encode
     the flat cell index and asserts the gather returns exactly the 37
     `ACTION_CELL_INDEXES` cells. This is the gate that would have caught the
     all-zeros gather-index bug.
  3. **masked softmax** — forbidden actions ≈ 0, distribution sums to 1, Hold is
     always sampleable (even when every action incl. its own column is masked),
     and a mostly-masked row of large equal logits stays finite.

## tfjs forward-parity dump — NOT included (and why)

A *true* numerical parity test would build the TF.js `v4.ts` model with a fixed
set of weights, dump its forward outputs for a fixed input, then assert the Rust
`ActorCriticV4` reproduces them to epsilon. This is **not feasible here without
standing up the full tfjs stack**, so per the task it is intentionally omitted
rather than faked. The blockers:

- It needs a **weight bridge** between burn's record format and tfjs
  `LayersModel` weights: matching every parameter by name AND by burn↔tfjs
  layout convention (e.g. `Linear` kernel transpose, attention head packing,
  orthogonal-init scorers, RMSNorm scale). None of that mapping exists yet.
- The v4 reference (`packages/ppo/src/models/v4.ts`) constructs a multi-output
  Perceiver `LayersModel`; driving it headless requires a tfjs backend
  (`@tensorflow/tfjs-backend-cpu` / `-wasm`) wired into a small Node harness.
- `@tensorflow/tfjs` is present at the repo root `node_modules`, and `node` is
  available — so it *could* be done — but the missing piece is the weight bridge,
  which is a substantial, separately-reviewable workstream, not a test.

### How to produce the dump when the bridge lands

1. In a Node script under `packages/ppo_unknown` (which already imports the v4
   model), build `createV4Network()`, set fixed weights (e.g. seed every layer's
   kernel with a deterministic generator, or `setWeights` from a saved array),
   run a forward pass on a fixed board+mask, and write the per-head logits +
   value to `tests/fixtures/v4_parity.json` (input board, input mask, output
   logits `[43]`, output value).
2. Add a Rust loader that constructs `ActorCriticV4`, loads the SAME weights
   (via the burn↔tfjs bridge / a recorder file exported from the same fixed
   weights), forwards the same fixed input, and asserts equality to ~1e-4
   (looser than the round-trip epsilon — different math libraries, summation
   order, and the WebGPU vs. tfjs-cpu kernels differ at the ULP level).
3. Gate it behind a `cfg(feature = "tfjs_parity")` or skip-if-missing-fixture so
   the default `cargo test` does not require the fixture to exist.

Do not weaken this to a finiteness check — if the fixture is absent the parity
test should be skipped (documented), never silently pass on garbage.
