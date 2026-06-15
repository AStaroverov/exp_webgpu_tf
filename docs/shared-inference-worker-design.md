# Shared Inference Worker for `ppo_unknown`

> Implementation-ready design. Encapsulate all PPO inference networks behind **one
> shared worker** that (1) maximizes batching across actors over a short flush window,
> and (2) hides the ML framework behind a blackbox seam so a future migration off
> TensorFlow.js is a single-file change. Source-of-truth files are referenced inline;
> do not invent APIs that contradict them.

---

## 1. Goal & motivation

Two goals drive this work:

1. **Max batching over a short window.** Today every tank decision is its own `B=1`
   forward pass on a per-worker `tf.LayersModel`. Four actor workers
   (`CONFIG.workerCount = 4`) each hold their own WebGPU context and run independent
   `B=1` inferences. The GPU is the bottleneck (per-decision latency is dominated by the
   `apply` + GPU→CPU readback roundtrip, ~7–30 ms), and `B=1` wastes almost all of it.
   Collecting concurrent requests across **all tanks and all actors** into one `B=N`
   forward pass amortizes the fixed per-call cost and frees three of the four WebGPU
   contexts.

2. **A blackbox seam for framework migration.** The act path is already ~90% generic
   Float32Array math (`packages/ppo/src/core/train.ts:169-194`); only three calls touch
   `tf` (`tf.tidy`, `policyNetwork.apply`, `tensor.data()`/`dispose`). We want a narrow
   `InferenceEngine` interface — `Float32Array` in, raw logits `Float32Array` out — so
   that swapping TF.js for ONNX Runtime Web / a custom WebGPU kernel touches exactly one
   implementation file and nothing above it.

**Why a shared worker (not a shared model in each actor):** batching requires a single
process that *sees* requests from all actors. With per-worker models there is no point
where cross-actor requests meet. One worker that owns the networks is the only place a
cross-actor batch can form. It also collapses 4 WebGPU contexts (one per actor) into 1,
and makes weight reloads a single load instead of four.

---

## 2. Current state — how inference works today

### 2.1 Actors and the decision trigger

- **Spawning** — `packages/ppo_unknown/src/entry/index.ts:17-26` spawns
  `CONFIG.workerCount` (= 4) `ActorWorker.ts` instances, each `{ type: "module" }`.
  Learner workers spawn 1 s later.
- **Per-worker TF init** — `packages/ppo_unknown/src/entry/ActorWorker.ts:16` calls
  `initTensorFlow("webgpu")`, so **each** actor holds its own WebGPU backend
  (`packages/ppo/src/infra/initTensorFlow.ts:5-29`). Custom layers are registered via the
  side-effect import `import "../models/createUnknownNetworks.ts"` at `ActorWorker.ts:6`.
- **Decision driver** — `packages/ppo_unknown/src/env/createPolicyDriverSystem.ts:44-89`
  (`updatePolicyDriver`) runs once per game tick in `SystemGroup.Before`. It queries all
  tanks (`Tank + Vehicle + VehicleController + Children`), early-exits if none satisfy
  `needsDecision(eid)`, snapshots the board once per tick
  (`snapshotUnknownBoard()`, line 64), then for **each** deciding tank calls
  `Promise.resolve(agent.decide())` (line 73), tracking it in `pending`/`inFlight` Sets.
  `drain()` (line 86-89) does `await Promise.all(pending)` — **the loop blocks every tick
  until all in-flight decisions settle** (`packages/ppo/src/core/EpisodeManager.ts:42-54`).

### 2.2 Networks and weight sourcing

- **Policy network** — `packages/ppo_unknown/src/env/UnknownAgent.ts:32-70`. A
  module-scoped `sharedNetwork` / `sharedNetworkPromise` is refreshed by
  `refreshSharedNetwork()` (loads `getNetwork(Model.Policy, CONFIG.savePath)` from
  IndexedDB via `patientAction`, disposes the previous model). `UnknownAgent.sync()`
  (line 68) is called **once per episode** by the episode manager.
- **Frozen opponent** — `packages/ppo_unknown/src/env/FrozenAgent.ts:27-47`. Same pattern,
  but `getRandomHistoricalNetwork()` re-rolls a random historical snapshot per episode.
- **Per-episode sync barrier** — `packages/ppo_unknown/src/agents/EpisodeManager.ts:91-95`
  `awaitAgentsSync()` does `Promise.all([UnknownAgent.sync(), FrozenAgent.sync()])` before
  the game loop runs.

### 2.3 The act path (where batching/seam live)

- `decide()` — `UnknownAgent.ts:105-152` builds `createInputTensors([state])`
  (`packages/ppo_unknown/src/state/InputTensors.ts:32-50`) → a `B=1` batch of two tensors:
  board `tf.tensor4d([B, 11, 11, 19])` and content mask `tf.tensor2d([B, 121])`. It calls
  `batchActAsync(network, input, [mask], options)` and destructures the single result, then
  `memory.addFirstPart(state, result.actions, result.logits, result.logProb, mask)`.
- `batchActAsync` — `packages/ppo/src/core/train.ts:155-195`. The **only** TF-coupled lines
  are `tf.tidy(() => parsePolicyOutput(policyNetwork.apply(inputTensors)))` (160-162) and
  `await Promise.all(heads.map((h) => h.data()))` + `h.dispose()` (164-165). Everything from
  line 169 onward is generic Float32Array scatter + `sampleCategorical` + `arrayHealthCheck`,
  producing `BatchActResult = { actions, logits, logProb }` (line 158 type).
- **Mask convention** — flat `Float32Array`, `0 = allowed`, `MASK_NEG = -1e9 = forbidden`
  (`train.ts:16`), applied additively inside `sampleCategorical` (`train.ts:183`).
- **Action layout** — single flat categorical head, `ACTION_DIM_TOTAL = 43`
  (`packages/ppo_unknown/src/consts.ts`): `[0]` Hold, `[1..6]` MoveStep, `[7..42]` Fire.
  Board is `BOARD_SIZE = 11*11*19 = 2299` floats (`packages/ppo_unknown/src/state/board.ts`).

### 2.4 Channels and batching points that exist today

- **`createChannel`** — `lib/channles.ts:4-41`. Dual-layer: local RxJS `Subject` +
  `BroadcastChannel`. **Critical for this design:** `response(cb)` (lines 31-39) does
  `response.next(res); crossResponse.postMessage(res)` — i.e. it **re-broadcasts every
  response to every subscriber**, and `await cb(data)` couples exactly one `Res` to one
  `Req`. With N actors, every actor would deserialize every other actor's response, and the
  1-req↔1-resp coupling fights batching. **This makes `createChannel` the wrong transport for
  the hot inference path** (it stays the right transport for the low-rate control plane).
- **Existing batching** — none at inference time. Cross-actor: none (independent episodes).
  Cross-tank within a tick: all `decide()` promises fire before `Promise.all(pending)`, but
  each is still a separate `B=1` forward pass. The only real batching today is **training**
  (`packages/ppo/src/learner/createPolicyLearnerAgent.ts`, mini-batches of 64 from 1024).
- **Control-plane channels** — `packages/ppo/src/core/channels.ts`
  (`agentSampleChannel`, `episodeSampleChannel`, `learnProcessChannel`, `queueSizeChannel`,
  `modelSettingsChannel`) and `packages/ppo_unknown/src/curriculumChannel.ts`. These run at
  episode rate and are fine on `BroadcastChannel`.

---

## 3. Target architecture

One `SharedInferenceWorker` owns every network. Each actor worker talks to it over a
**dedicated `MessagePort`** (point-to-point, transferable handoff) for the hot path. The
low-rate control plane stays on `createChannel`/`BroadcastChannel`.

```
                          packages/ppo_unknown/src/entry/index.ts
                          (spawns SharedInferenceWorker FIRST, then actors;
                           creates one MessageChannel per actor and hands
                           port1→inference, port2→actor)

   ActorWorker #0 ─┐                                            ┌─ control plane (createChannel / BroadcastChannel)
   ActorWorker #1 ─┤   per-actor MessagePort (Float32Array)     │   inferenceReloadChannel  (learner → worker)
   ActorWorker #2 ─┤  ───────────────────────────────────────► │   curriculumStateChannel  (unchanged)
   ActorWorker #3 ─┘   req: {model, board, mask, greedy, ver}   │   agent/episodeSampleChannel (unchanged)
        ▲                resp: {logits, actions, logProb, ver}  │
        │                                                       ▼
        │            ┌──────────────────── SharedInferenceWorker ─────────────────────┐
        │            │  InferenceClient ports[]   per-model request queues:            │
        │            │      policy_main   ──► Batcher ──► stack (memcpy into scratch)   │
        │            │      policy_frozen ──►            │                             │
        │            │                                   ▼                             │
        └────────────┤  scatter results back  ◄──── sampleBatch (generic, JS)          │
        per-port      │      (per-port reply)         ▲                                │
        reply         │                                │ raw logits Float32Array[n*43] │
                      │                       InferenceEngine.infer(model, stacked, n)  │
                      │                                │                                │
                      │                       TfInferenceEngine (ONLY file importing tf)│
                      │                         initTensorFlow("webgpu") + apply + data │
                      └─────────────────────────────────────────────────────────────────┘

   Learner saves weights → IndexedDB (Transfer.ts) → emits inferenceReloadChannel
   → worker.loadWeights() into staging slot → atomic ref swap between flushes
```

Key consequences:

- **`ActorWorker.ts` no longer calls `initTensorFlow("webgpu")`** — 4 WebGPU contexts →
  1, all in the inference worker. Actors hold no `tf` model.
- **`UnknownAgent`/`FrozenAgent` no longer own networks.** `decide()` becomes a port
  round-trip through `InferenceClient`. `refreshSharedNetwork`/`sync()` move to the worker.
- **Batches form naturally:** one actor tick already emits N-tank requests before it drains;
  4 actors overlap, so tens of requests can be in flight inside one window.

---

## 4. The blackbox seam — `InferenceEngine`

The seam is `Float32Array` in, **raw logits** `Float32Array` out. No `tf` type crosses it,
and **no sampling** happens inside it (see §8 for why sampling must stay in JS).

```ts
// packages/ppo/src/inference/types.ts
export type ModelKind = "policy_main" | "policy_frozen";

// packages/ppo/src/inference/InferenceEngine.ts
export interface InferenceEngine {
  /** Load (or reload) the weights for one model into a staging slot, then swap. */
  loadWeights(model: ModelKind): Promise<void>;

  /** Exp-iteration / version of the currently-served weights for `model`. */
  version(model: ModelKind): number;

  /**
   * Forward pass only. `stacked` is `n` boards concatenated, row-major,
   * length === n * BOARD_SIZE (2299). Masks are NOT passed here — masking is
   * additive at sample time, above the seam. Returns concatenated raw logits,
   * length === n * ACTION_DIM_TOTAL (43). No tensors, no sampling, no disposal
   * leaking to the caller.
   */
  infer(model: ModelKind, stacked: Float32Array, n: number): Promise<Float32Array>;

  dispose(): void;
}
```

**Generic (above the seam, framework-neutral):** request queues, the 50–100 ms window,
stacking via memcpy, the `sampleBatch` tail (mask application, `sampleCategorical`,
`logProb` accumulation, `arrayHealthCheck`, `BatchActResult` assembly), port reply, weight
double-buffer/swap logic, version tagging.

**TF-specific (below the seam — `TfInferenceEngine.ts` only):** `initTensorFlow("webgpu")`,
holding the `tf.LayersModel`, `tf.tidy` + `policyNetwork.apply`, `parsePolicyOutput`
(`train.ts:171` region — relocated, not rewritten), `tensor.data()` async readback +
`dispose`, `getNetwork`/`getNetworkExpIteration`/`disposeNetwork` (relocated from
`UnknownAgent.ts`/storage), `onReadyRead()` to gate disposal.

**How to swap backends later:** write a new class implementing `InferenceEngine`
(e.g. `OnnxInferenceEngine`). It must:
1. reshape `stacked` into `[n, 11, 11, 19]` + build the `[n, 121]` content mask the model
   expects (the content mask is **structural**, derived from the board — it is *not* the
   action mask; reconstruct it the same way `createInputTensors` does today, or pass it as
   a second stacked buffer if a future engine needs it precomputed),
2. run the forward pass,
3. return raw logits `Float32Array` of length `n * 43`.
Nothing above the seam changes; the Batcher, client, worker, and agents are untouched.

> Note on the content mask: today `createInputTensors` produces a 2nd tensor `[B, 121]`.
> Keep that derivation **inside** `TfInferenceEngine` (it is TF input plumbing), or, if a
> future engine wants it precomputed on CPU, widen the seam minimally to
> `infer(model, board, contentMask, n)` — board and contentMask both flat `Float32Array`.
> Prefer keeping it inside the engine until a second engine actually needs otherwise
> (CLAUDE.md: don't build speculative generality).

---

## 5. Request/response protocol

### 5.1 Message shapes

All wire types contain **only** `Float32Array` + numbers — never `tf`.

```ts
// packages/ppo/src/inference/types.ts
export type InferRequest = {
  reqId: number;          // monotonic per-port; correlates resp to the awaiting promise
  model: ModelKind;       // selects the queue + weights; cannot mix within a batch row-set
  board: Float32Array;    // length BOARD_SIZE (2299); the actor's copied-out snapshot
  mask: Float32Array;     // length ACTION_DIM_TOTAL (43); 0 allowed / -1e9 forbidden
  greedy: boolean;        // resolved per-row in the sampler; may freely mix within a batch
  reqVersion: number;     // version the caller believes is current (staleness diagnostics)
};

export type InferResponse = {
  reqId: number;          // echoes the request
  actions: Float32Array;  // one action index per head (here length 1: the flat head)
  logits: Float32Array;   // raw concatenated logits, length 43
  logProb: number;        // summed log-prob of the chosen action(s)
  modelVersion: number;   // version that ACTUALLY served this result (for memory tagging)
};
```

Setup messages on the structural channel (worker init, port handoff):

```ts
// index.ts → inference worker (port1)
inferenceWorker.postMessage({ type: "port" }, [ch.port1]);
// index.ts → actor worker (port2)
actorWorker.postMessage({ type: "inferencePort", port: ch.port2 }, [ch.port2]);
```

### 5.2 Correlation

`reqId` is a monotonic counter **per `InferenceClient`** (per actor). The client keeps a
`Map<reqId, {resolve, reject}>`; the worker echoes `reqId` in `InferResponse`; the client
resolves the matching promise and deletes the entry. Because each actor has a dedicated
port, there is **no cross-actor fan-out and no global correlation table** — the worker
already knows which port a request arrived on and replies on that same port.

### 5.3 Transfer strategy & allocation discipline

- **v1 — `postMessage` transferables (ship this).** The actor must copy `state.board` out
  of the reused `ensureUnknownInputBoard(world).board.getBatch(eid)` scratch *anyway*
  (the next tick overwrites it — `InputArrays.ts`). Make that forced copy the transfer
  buffer and hand its backing `ArrayBuffer` over zero-copy:
  ```ts
  port.postMessage(req, [req.board.buffer, req.mask.buffer]);
  ```
  No extra allocation beyond the copy that already had to happen. The response
  (`logits`/`actions` Float32Arrays) is freshly allocated in `sampleBatch` and transferred
  back the same way.
- **v2 — SharedArrayBuffer ring (optional, gated behind profiling).** A SAB ring per port
  removes the postMessage serialization hop entirely, but requires COOP/COEP headers and
  careful index/fence discipline. Per CLAUDE.md ("profile before optimizing", worst-frame
  percentiles), do **not** build this until v1 is profiled and shown to be transport-bound.
- **Worker-side scratch.** The Batcher owns a single preallocated `boardScratch` of
  `MAX_BATCH * BOARD_SIZE` floats and reuses it every flush (length-reset, not realloc).
  Stacking is a straight `boardScratch.set(req.board, row * BOARD_SIZE)` memcpy. No ragged
  padding ever — every board is exactly 2299 floats, every mask exactly 43. This is the
  CLAUDE.md hot-loop rule (reused scratch buffers, allocation-free, plain `for`).

---

## 6. Batching design

Per-model FIFO queues: `policy_main` and `policy_frozen` batch **separately** (different
weights → different `apply`). `greedy` does **not** split a batch — it only affects sampling,
which is per-row on CPU, so a single forward pass may mix greedy and non-greedy rows.

### 6.1 Flush triggers

A queue flushes when **either** condition fires (whichever is first):

1. **Size (primary):** `queue.length >= MAX_BATCH`.
2. **Window ceiling:** a timer set to **60 ms** that starts on the **first enqueue after a
   flush** (not on every enqueue). On fire, flush whatever is queued.

```ts
function enqueue(model: ModelKind, req: InferRequest, port: MessagePort) {
  const q = queues[model];
  q.items.push({ req, port });
  if (q.items.length >= MAX_BATCH) { flush(model); return; }
  if (q.timer == null) {
    q.timer = setTimeout(() => flush(model), WINDOW_MS); // WINDOW_MS = 60
  }
}
```

### 6.2 Stack → infer → scatter

```ts
async function flush(model: ModelKind) {
  const q = queues[model];
  if (q.timer != null) { clearTimeout(q.timer); q.timer = null; }
  const items = q.items; q.items = []; // swap out; new arrivals start a fresh window
  if (items.length === 0) return;

  const n = items.length;
  for (let i = 0; i < n; i++) boardScratch.set(items[i].req.board, i * BOARD_SIZE);

  const ver = engine.version(model);
  // engine captured its `net` local on entry; a weight swap mid-flight is safe (see §7)
  const logitsFlat = await engine.infer(model, boardScratch.subarray(0, n * BOARD_SIZE), n);

  // GENERIC tail — sampleBatch(): extracted verbatim from batchActAsync (train.ts:169-194)
  for (let i = 0; i < n; i++) {
    const raw = logitsFlat.subarray(i * ACTION_DIM_TOTAL, (i + 1) * ACTION_DIM_TOTAL);
    const { actions, logits, logProb } = sampleBatchRow(raw, items[i].req.mask, items[i].req.greedy);
    const resp: InferResponse = { reqId: items[i].req.reqId, actions, logits, logProb, modelVersion: ver };
    items[i].port.postMessage(resp, [resp.logits.buffer, resp.actions.buffer]);
  }
}
```

`sampleBatchRow` is the inner body of the `for (i)` loop in `train.ts:170-192`:
store raw logits, apply the additive mask via `sampleCategorical`, accumulate `logProb`,
`arrayHealthCheck`. **Extract it as a pure function first** (migration step 1) so training
and the worker share one copy.

### 6.3 Latency / throughput tradeoff vs the game tick

The actor **blocks on `drain()` every tick** (`EpisodeManager.runGameLoop`,
`createPolicyDriverSystem.drain`). Therefore:

- A long window on a blocking drain **serializes actors and starves the queue** — tick
  N+1's request cannot be enqueued until tick N resolves, so a 100 ms window would mostly
  time out near-empty. **The window is a tail-latency ceiling, not the feed mechanism.**
- The batch is fed by **intra-tick concurrency**: `updatePolicyDriver` fires *all* deciding
  tanks' promises before `Promise.all(pending)`. One tick from one actor already emits
  N-tank requests; × `workerCount` (4) overlapping actors = tens of concurrent requests per
  window. **`MAX_BATCH` is the primary flush path.**
- **If the batcher starves under real load, the lever is more actors**
  (`CONFIG.workerCount` 8–16), not a longer window — more actors = more independent ticks
  overlapping = bigger natural batches without adding per-decision latency.

**Adaptive window (optional refinement, not load-bearing):** for the genuine single-actor
debug case (1 worker, few tanks), shrink `WINDOW_MS` toward the GPU floor so a lone actor
isn't penalized waiting for a batch that will never fill. Keep this strictly as a debug
ergonomics lever; production tuning is `MAX_BATCH` + `workerCount`.

Recommended starting constants: `WINDOW_MS = 60`, `MAX_BATCH = 64` (matches the training
mini-batch the network already runs comfortably; tune against worst-frame percentiles).

---

## 7. Weight updates mid-stream

The shared worker must accept new policy weights from the learner **without corrupting an
in-flight batch**.

1. **Learner side (unchanged plumbing):** trainer saves to IndexedDB via `saveNetworkToDB`
   (`packages/ppo/src/models/Transfer.ts:15-43`), then emits on a **new** low-rate
   `inferenceReloadChannel` (a `createChannel` control-plane message — fine at this rate).
   For the frozen pool, the per-episode reroll instruction also rides the control plane.
2. **Worker side — double buffer + atomic swap between flushes.** `loadWeights(model)`
   loads into a **staging slot** (`getNetwork` for `policy_main`,
   `getRandomHistoricalNetwork` for `policy_frozen`). The live reference is swapped in **one
   synchronous assignment**, and only **between flushes** (never mid-`infer`).
3. **Safe disposal invariant (the load-bearing correctness argument):** an in-flight `infer`
   already **captured its `net` local** when the flush began, so it completes against the
   old weights with no torn read. The swap merely repoints the field for the *next* flush.
   `disposeNetwork(prev)` runs only **after** the in-flight forward pass settles — gate it
   on the per-model in-flight counter (or `await onReadyRead()` as `Transfer.ts:16` already
   does before saves) before disposing.
4. **Version correctness:** `engine.version(model)` is captured at flush start and stamped
   onto every `InferResponse.modelVersion`. The actor tags `memory.addFirstPart` with the
   version that **actually served** the sample — correct regardless of when a reload landed.
   `getVersion()` on the agents is served from the latest `modelVersion` the client has seen.

**Frozen-pool LRU thrashing (must mitigate):** if every actor rerolls a *different* frozen
version per episode, `policy_frozen` batches fragment per version (each version = different
weights = its own forward pass). Mitigation: **cap the number of distinct frozen versions
loaded at once** (small LRU, e.g. 2–4 slots keyed by version) and accept that frozen batches
(greedy eval, latency-tolerant) run smaller than `policy_main`. Coordinate rerolls so actors
prefer an already-loaded frozen version when curriculum allows.

---

## 8. Edge cases & invariants

- **Masks stay additive, above the seam.** Mask is `Float32Array(43)`, `0 = allowed`,
  `-1e9 = forbidden` (`train.ts:16`), applied inside `sampleCategorical` as
  `logits[k] + mask[k]` (`train.ts:183`). The engine returns **raw, unmasked** logits;
  masking and storage of raw logits for training happen in `sampleBatchRow` exactly as
  `train.ts:172-182` does today.
- **`sampleCategorical` MUST stay in JS — do not push it into the engine.** The TF GPU path
  (`sampleActionsFromLogits`, `train.ts:204-211`) uses `tf.multinomial(..., 1)`, which seeds
  per-call and is the source of the known seed bug — the act path deliberately reads logits
  back and samples on CPU. A future ONNX/PyTorch engine that "samples in-graph" would
  reintroduce that class of bug and break `logProb` consistency. The seam contract is **raw
  logits out**, full stop.
- **Greedy vs sampling.** `greedy` is per-row, resolved in `sampleBatchRow`; a batch may mix
  greedy (reference/frozen eval) and sampling (training) rows freely — it does not affect the
  forward pass, only the CPU tail. Greedy rows draw no RNG (argMax), so they stay reproducible.
- **Determinism caveat (document it).** Batching reorders `sampleCategorical`'s global-PRNG
  draws across actors, so fixed-seed reproducibility **diverges from today** — each draw is
  still individually correct and `logProb` still matches the chosen action, but the global RNG
  interleaving changes. **Greedy reference/frozen episodes remain reproducible** (no RNG).
- **B=1 vs B=N numerical parity.** The network must produce bit-comparable per-row logits at
  `B=N` vs `B=1`. v4 uses per-token LayerNorm and per-row attention (no batch-coupled ops like
  batchnorm), so this holds — **assert it in the engine parity test** (§10 step 2).
- **NaN / health.** `arrayHealthCheck(logits)` + `Number.isFinite(logProb)` run per row in
  `sampleBatchRow` (`train.ts:189`), throwing on bad values — preserved verbatim. The worker
  must catch a thrown row error and reject the matching `reqId` promise so the actor surfaces
  it (today an actor failure triggers `forceExitChannel`).
- **Frozen / opponent agents** are just `model: "policy_frozen"` requests through the same
  port and Batcher; no separate transport. The only new coordination the shared model forces
  is per-episode frozen-version selection on the control plane (§7).

---

## 9. File & module plan

**New — `packages/ppo/src/inference/`:**
- `types.ts` — `ModelKind`, `InferRequest`, `InferResponse`, wire constants.
- `InferenceEngine.ts` — the `InferenceEngine` interface (§4).
- `TfInferenceEngine.ts` — the **only** file importing `@tensorflow/tfjs`; relocates
  `initTensorFlow("webgpu")`, `getNetwork`/`getNetworkExpIteration`/`disposeNetwork`,
  `parsePolicyOutput`, `apply`, `data()`/`dispose`, `onReadyRead()`.
- `Batcher.ts` — per-model queues, window timer, `boardScratch`, stack/flush/scatter,
  `sampleBatchRow` (imported from `train.ts` after extraction).
- `InferenceClient.ts` — actor-side: holds the `MessagePort`, `reqId` counter, pending-promise
  map; exposes `infer(model, board, mask, greedy): Promise<InferResponse>`.

**New — `packages/ppo_unknown/src/entry/`:**
- `InferenceWorker.ts` — worker entry: side-effect import `../models/createUnknownNetworks.ts`
  (custom layers, as `ActorWorker.ts:6`), constructs `TfInferenceEngine` + `Batcher`, accepts
  ports, subscribes to `inferenceReloadChannel` + frozen-reroll control messages.

**New — channel:**
- `inferenceReloadChannel` in `packages/ppo/src/core/channels.ts` (or a new
  `packages/ppo_unknown/src/inferenceChannel.ts`) — low-rate `createChannel` for
  weight-reload + frozen-pick notifications.

**Changed:**
- `packages/ppo_unknown/src/entry/index.ts` — spawn `InferenceWorker` **first**; create one
  `MessageChannel` per actor; hand `port1`→inference, `port2`→actor (§5.1).
- `packages/ppo_unknown/src/entry/ActorWorker.ts` — **remove** `initTensorFlow("webgpu")`
  (frees 3 WebGPU contexts); receive the inference port and build an `InferenceClient`.
- `packages/ppo_unknown/src/env/UnknownAgent.ts` — `decide()` → `inferenceClient.infer(...)`;
  delete `sharedNetwork`/`refreshSharedNetwork`/local TF; `getVersion()` from `modelVersion`.
- `packages/ppo_unknown/src/env/FrozenAgent.ts` — same; frozen reroll moves to control plane.
- `packages/ppo/src/core/train.ts` — extract `sampleBatchRow` (the `for (i)` body, lines
  170-192) into a pure exported function; `batchActAsync` becomes `forward → sampleBatchRow`
  loop, behavior-identical. Used by both training and the Batcher.

**Unchanged:** `createPolicyDriverSystem.ts` pending/drain logic, `Memory.ts`,
`EpisodeManager` loop, `InputArrays.ts`/`InputTensors.ts` board layout, all metrics channels.

---

## 10. Migration steps

Each step is independently shippable and testable.

1. **Pure refactor — extract `sampleBatchRow` from `batchActAsync`** (`train.ts:170-192`).
   No behavior change. Test: existing training + actor paths green; `batchActAsync` output
   bit-identical on a fixed input. *Biggest single risk-reducer; lands before any worker code.*
2. **`InferenceEngine` + `TfInferenceEngine`** (no worker yet). Wire `batchActAsync` to call
   `TfInferenceEngine.infer` internally. **Parity test:** raw logits from `infer(model, stacked, n)`
   match the current `policyNetwork.apply` path **bit-for-bit** at `B=1` and at `B=N`
   (validates §8 numerical parity).
3. **`Batcher`** (in-process, no worker). Unit-test: scatter ordering (row i ↔ reqId i),
   mixed greedy/non-greedy in one batch, window-timeout flush vs size flush, scratch reuse
   (no per-flush allocation).
4. **`InferenceWorker` + `InferenceClient` + port handshake**, behind a feature flag
   (`CONFIG.sharedInference`). Actors can run either local-TF (old) or client (new). Test:
   one actor, flag on, episodes complete; logits/actions match local-TF within the
   determinism caveat (greedy rows must match exactly).
5. **Flip agents to the client; remove actor `initTensorFlow`.** Confirm only the inference
   worker creates a WebGPU context. Run all 4 actors; verify batch sizes > 1 under load.
6. **Wire `inferenceReloadChannel` + frozen LRU.** Test mid-stream reload: weights swap with
   no torn read, `modelVersion` advances on responses, in-flight batch completes on old
   weights, old model disposed only after settle.
7. **Profile worst-frame percentiles** (post-JIT warmup): per-decision latency, achieved batch
   sizes, GPU utilization. Tune `WINDOW_MS`/`MAX_BATCH`/`workerCount`.
8. **(Optional) SAB ring transport** only if step 7 shows the path is transport-bound.

---

## 11. Risks & open questions

- **Window starvation under blocking drain.** Mitigated by relying on intra-tick + cross-actor
  concurrency and `MAX_BATCH` as the primary trigger (§6.3). Open: the exact `WINDOW_MS`/
  `MAX_BATCH`/`workerCount` operating point — resolve empirically in step 7.
- **Frozen-version fragmentation.** The frozen LRU cap (§7) bounds it but means frozen batches
  run smaller. Open: whether to coordinate actor rerolls toward shared versions, or accept the
  smaller batches (frozen eval is latency-tolerant).
- **Determinism divergence.** Fixed-seed runs will not reproduce today's exact trajectories due
  to cross-actor PRNG interleaving (§8). Acceptable for training; flag it for anyone relying on
  seeded reproducibility. Greedy paths unaffected.
- **B=N parity assumption.** Holds for v4 (per-token LayerNorm, per-row attention), but any
  future architecture with batch-coupled ops (batchnorm, batch-wide reductions) would break it.
  The step-2 parity test guards this.
- **Single point of failure.** One inference worker now gates all actors; a crash stalls
  everything. Hook it to `forceExitChannel` (`packages/ppo/src/infra/channels.ts`) for the
  existing reload-on-error recovery, same as today's per-actor failures.
- **Port lifecycle.** Actors are long-lived, so ports are set up once at handshake. Open: if an
  actor worker ever restarts, `index.ts` must re-create and re-hand a fresh `MessageChannel`.
- **Content-mask seam width.** Kept inside `TfInferenceEngine` for now (§4). Open: if a second
  engine wants it precomputed, widen the seam to `infer(model, board, contentMask, n)` — defer
  until a real second engine exists (no speculative generality).
