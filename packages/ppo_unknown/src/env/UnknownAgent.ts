/**
 * UnknownAgent — the per-tank actor for ppo_unknown. The analogue of tanks'
 * `CurrentActorAgent`, simplified for the chess-like board + decision-point cadence:
 *
 *   - holds an `AgentMemory<InputArrays>` filled with the two-phase API;
 *   - at each decision point (driven by `createPolicyDriverSystem`):
 *       1. close the PREVIOUS step  → updateSecondPart(reward, done)
 *       2. sample an action         → batchAct on the shared policy network
 *       3. enqueue it               → applyActionToGame
 *       4. open the new step        → addFirstPart(state, action, logits, logProb)
 *   - reward of a macro-action = real combat delta Φ_combat(s') − Φ_combat(s)
 *     (objective, never annealed) + potential-based approach shaping
 *     γ·Φ_appr(s') − Φ_appr(s) scaled by `shapingWeight` (annealable to zero).
 *
 * Network weights are pulled from IndexedDB storage by a worker-shared updater
 * (the learner writes them there); `sync()` refreshes them between episodes.
 */

import * as tf from "@tensorflow/tfjs";
import { AgentMemory, AgentMemoryBatch } from "../../../ppo/src/memory/Memory.ts";
import { batchActAsync } from "../../../ppo/src/core/train.ts";
import { Model } from "../../../ppo/src/models/def.ts";
import { getNetwork, disposeNetwork } from "../../../ppo/src/models/storage.ts";
import { getNetworkExpIteration } from "../../../ppo/src/models/networkMeta.ts";
import { patientAction } from "../../../ppo/src/utils/patientAction.ts";
import { getTankHealth } from "../../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts";
import { CONFIG } from "../config.ts";
import { createInputTensors } from "../state/InputTensors.ts";
import { InputArrays, prepareInputArrays } from "../state/InputArrays.ts";
import { calculateActionReward, calculateShapingPotential } from "../reward/calculateReward.ts";
import { applyActionToGame } from "./applyActionToGame.ts";
import { computeActionMaskWithPrior } from "./computeActionMask.ts";

// ── Worker-shared policy network updater ─────────────────────────────────────
let sharedNetwork: tf.LayersModel | undefined;
let sharedNetworkPromise: Promise<tf.LayersModel> | undefined;

async function refreshSharedNetwork(): Promise<tf.LayersModel> {
  const prev = sharedNetworkPromise;
  sharedNetworkPromise = patientAction(() => getNetwork(Model.Policy, CONFIG.savePath));
  const next = await sharedNetworkPromise;
  if (sharedNetwork && sharedNetwork !== next) disposeNetwork(sharedNetwork);
  sharedNetwork = next;
  await prev?.catch(() => undefined);
  return next;
}

// ── Inference-mode override (vis tab) ────────────────────────────────────────
// When set, overrides the train-based sampling choice in `decide()`. Only the
// main tab's dashboard toggles this; actor workers never call it, so training
// keeps stochastic sampling.
let greedyOverride: boolean | undefined;

export function setGreedyInference(value: boolean | undefined): void {
  greedyOverride = value;
}

export class UnknownAgent {
  private memory = new AgentMemory<InputArrays>();
  private opened = false;
  /** Combat potential at the open step (real reward = its per-action delta). */
  private prevCombat = 0;
  /** Approach-shaping potential at the open step (annealed γ-potential). */
  private prevShaping = 0;

  constructor(
    public readonly tankEid: number,
    public readonly train: boolean,
    public readonly shapingWeight: number,
  ) {}

  /** Pull the latest policy weights (called once per episode by the manager). */
  static sync(): Promise<unknown> {
    return refreshSharedNetwork();
  }

  getVersion(): number {
    return sharedNetwork != null ? getNetworkExpIteration(sharedNetwork) : 0;
  }

  /**
   * Reward for the macro-action being closed, evaluated at the current state:
   *   - combat: a REAL reward = Φ_combat(s') − Φ_combat(s) (objective, never annealed);
   *   - approach: potential-based shaping γ·Φ_appr(s') − Φ_appr(s), scaled by `shapingWeight`
   *     (policy-invariant, so annealing it to zero cannot change the optimum).
   * We do NOT force the strict Ng terminal Φ(terminal)=0 — that would inject a −Φ_last
   * spike on the step that already carries the terminal win/loss reward; treating the last
   * observed potential as terminal keeps the shaping bounded (≈ (γ−1)·Φ).
   */
  private closingReward(): number {
    const gamma = CONFIG.gamma(this.getVersion());
    const combatDelta = calculateActionReward(this.tankEid) - this.prevCombat;
    const shapingDelta =
      (gamma * calculateShapingPotential(this.tankEid) - this.prevShaping) * this.shapingWeight;
    return combatDelta + shapingDelta;
  }

  closeFinalStep(): void {
    if (!this.train || !this.opened) return;
    this.memory.updateSecondPart(this.closingReward(), true);
    this.opened = false;
  }

  getMemoryBatch(finalReward: number): undefined | AgentMemoryBatch<InputArrays> {
    return this.train ? this.memory.getBatch(finalReward) : undefined;
  }

  /** Sum of the last `n` recorded macro-action rewards (debug/visualizer only). */
  getRecentReward(n = 10): number {
    const rewards = this.memory.rewards;
    let sum = 0;
    for (let i = Math.max(0, rewards.length - n); i < rewards.length; i++) {
      sum += rewards[i];
    }
    return sum;
  }

  dispose(): void {
    this.memory.dispose();
  }

  /**
   * One decision step. `snapshotUnknownBoard` must already have filled this
   * tank's board row this tick (the driver does it once for all observers).
   */
  async decide(): Promise<void> {
    const network = sharedNetwork;
    if (network == null) return;

    // 1. Close the previous macro-action with its accumulated reward (real combat
    // delta + annealed approach potential — see `closingReward`).
    if (this.opened && this.train) {
      const done = getTankHealth(this.tankEid) <= 0;
      this.memory.updateSecondPart(this.closingReward(), done);
      if (done) {
        this.opened = false;
        return;
      }
    }

    // 2. Capture the state AND both opening potentials synchronously, then sample
    // asynchronously. The potentials are read here (not after the await) so their
    // timing matches the old sync driver — before this tick's `updateHitableSystem`
    // applies damage — keeping the reward baseline consistent with the closing read.
    const state = prepareInputArrays(this.tankEid);
    const mask = computeActionMaskWithPrior(this.tankEid);
    const nextCombat = calculateActionReward(this.tankEid);
    const nextShaping = calculateShapingPotential(this.tankEid);
    const options = { greedy: greedyOverride ?? !this.train };
    const input = createInputTensors([state]);
    let result;
    try {
      [result] = await batchActAsync(network, input, [mask], options);
    } finally {
      input.forEach((t) => t.dispose());
    }

    // The tank may have died while inference was in flight (vis fire-and-forget;
    // the actor drains same-tick so this never trips there).
    if (getTankHealth(this.tankEid) <= 0) {
      this.opened = false;
      return;
    }

    // 3. Enqueue it into the game.
    applyActionToGame(this.tankEid, result.actions);

    // 4. Open the new step.
    if (this.train) {
      this.memory.addFirstPart(state, result.actions, result.logits, result.logProb, mask);
      this.prevCombat = nextCombat;
      this.prevShaping = nextShaping;
      this.opened = true;
    }
  }
}
