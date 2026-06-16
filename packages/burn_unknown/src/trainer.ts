/**
 * burnTrainer — the process-wide single Burn `V4Trainer` instance, the SINGLE-THREAD
 * replacement for ppo_unknown's worker split (ActorWorker inference + LearnerPolicy/
 * Value training + IndexedDB weight transfer all collapse into one in-memory object).
 *
 * In the real tfjs flow:
 *   - actors call `batchActAsync(policyNetwork, …)` for inference, reading weights
 *     the learners wrote to IndexedDB;
 *   - learners call `trainPolicyNetwork` / `trainValueNetwork` (PPO + value clip) on
 *     batched samples and save weights back.
 *
 * Here `V4Trainer` owns BOTH the policy+value network and the AdamW optimiser pair in
 * one wasm object, so:
 *   - `act(board, mask, greedy)` IS the inference call (returns [action, logProb, value]);
 *   - `update(...)` IS the whole learner (Retrace + clipped PPO + value loss + KL-adaptive
 *     LR), run in-process on the flat rollout the same object just sampled.
 * No weight transfer, no storage, no version metadata — the live object is always the
 * latest policy. `getVersion()` is just the count of completed `update()`s.
 */

import init, { V4Trainer, type V4IterStats } from "../../burn_rl/pkg/burn_rl.js";

export type { V4IterStats };

let trainer: V4Trainer | undefined;
let version = 0;
let initPromise: Promise<V4Trainer> | undefined;

const SEED = 42;

/** Initialise wasm + the V4Trainer once; subsequent calls return the same instance. */
export function ensureTrainer(): Promise<V4Trainer> {
  if (initPromise) return initPromise;
  initPromise = (async () => {
    await init();
    trainer = await V4Trainer.create(SEED);
    return trainer;
  })();
  return initPromise;
}

export function getTrainer(): V4Trainer | undefined {
  return trainer;
}

// ── Serialization lock ───────────────────────────────────────────────────────
// `V4Trainer` is a single wasm object whose `act`/`update` are `async fn(&mut self)`:
// wasm-bindgen holds the object's borrow for the WHOLE future, so a second call that
// starts before the first resolves panics with "recursive use of an object ... unsafe
// aliasing in rust". The policy driver kicks off `decide()` (→ `act`) for several tanks
// in the SAME tick, so those calls overlap. Funnel every wasm call through one promise
// chain so they run strictly one-at-a-time (the tfjs path tolerated concurrency; this
// one must not).
let lock: Promise<unknown> = Promise.resolve();

function withTrainerLock<T>(fn: () => Promise<T>): Promise<T> {
  const run = lock.then(fn, fn);
  // Keep the chain alive but swallow errors on the gate so one failure doesn't poison
  // every subsequent call; the real result/rejection is returned to the caller via `run`.
  lock = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

/** Serialized inference: `[action, logProb, value]` for a single observation. */
export async function trainerAct(
  board: Float32Array,
  mask: Float32Array,
  greedy: boolean,
): Promise<Float32Array> {
  const t = await ensureTrainer();
  return withTrainerLock(() => t.act(board, mask, greedy));
}

/** Iteration counter — the analogue of the tfjs network's experience-iteration meta. */
export function getTrainerVersion(): number {
  return version;
}

export function bumpTrainerVersion(): void {
  version += 1;
}

export const ACTION_DIM = 43;

// ── Inference-mode override (vis tab) ────────────────────────────────────────
// Mirrors UnknownAgent.setGreedyInference: when set, overrides the train-based
// sampling choice so the rendered episode can be watched greedily.
let greedyOverride: boolean | undefined;

export function setGreedyInference(value: boolean | undefined): void {
  greedyOverride = value;
}

export function getGreedyOverride(): boolean | undefined {
  return greedyOverride;
}

/** Run one PPO update over a flat rollout batch and bump the version. */
export async function trainerUpdate(
  boards: Float32Array,
  masks: Float32Array,
  actions: Int32Array,
  oldLogp: Float32Array,
  rewards: Float32Array,
  dones: Float32Array,
  values: Float32Array,
): Promise<V4IterStats> {
  const t = await ensureTrainer();
  const stats = await withTrainerLock(() =>
    t.update(boards, masks, actions, oldLogp, rewards, dones, values),
  );
  bumpTrainerVersion();
  return stats;
}
