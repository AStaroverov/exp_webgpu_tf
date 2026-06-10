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
 *   - reward of a macro-action = Δ(action potential) over its duration + time cost.
 *
 * Network weights are pulled from IndexedDB storage by a worker-shared updater
 * (the learner writes them there); `sync()` refreshes them between episodes.
 */

import * as tf from "@tensorflow/tfjs";
import { AgentMemory, AgentMemoryBatch } from "../../../ppo/src/memory/Memory.ts";
import { batchAct } from "../../../ppo/src/core/train.ts";
import { Model } from "../../../ppo/src/models/def.ts";
import { getNetwork, disposeNetwork } from "../../../ppo/src/models/storage.ts";
import { getNetworkExpIteration } from "../../../ppo/src/models/networkMeta.ts";
import { patientAction } from "../../../ppo/src/utils/patientAction.ts";
import { getTankHealth } from "../../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts";
import { CONFIG } from "../config.ts";
import { createInputTensors } from "../state/InputTensors.ts";
import { InputArrays, prepareInputArrays } from "../state/InputArrays.ts";
import { calculateActionReward } from "../reward/calculateReward.ts";
import { applyActionToGame } from "./applyActionToGame.ts";
import { computeActionMask } from "./computeActionMask.ts";

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

export class UnknownAgent {
  private memory = new AgentMemory<InputArrays>();
  private opened = false;
  private prevPotential = 0;

  constructor(
    public readonly tankEid: number,
    public readonly train: boolean,
    /** Dense-shaping anneal factor for this episode (see `getShapingWeight`). */
    public readonly shapingWeight: number,
  ) {}

  /** Pull the latest policy weights (called once per episode by the manager). */
  static sync(): Promise<unknown> {
    return refreshSharedNetwork();
  }

  getVersion(): number {
    return sharedNetwork != null ? getNetworkExpIteration(sharedNetwork) : 0;
  }

  closeFinalStep(): void {
    if (!this.train || !this.opened) return;
    const delta = (calculateActionReward(this.tankEid) - this.prevPotential) * this.shapingWeight;
    this.memory.updateSecondPart(delta, true);
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
  decide(): void {
    if (sharedNetwork == null) return;

    // 1. Close the previous macro-action with its accumulated reward.
    if (this.opened && this.train) {
      const delta = (calculateActionReward(this.tankEid) - this.prevPotential) * this.shapingWeight;
      const done = getTankHealth(this.tankEid) <= 0;
      this.memory.updateSecondPart(delta, done);
      if (done) {
        this.opened = false;
        return;
      }
    }

    // 2. Sample an action for the current state.
    const state = prepareInputArrays(this.tankEid);
    const mask = computeActionMask(this.tankEid);
    const options = this.train ? { greedy: false } : { greedy: true };
    const input = createInputTensors([state]);
    const [result] = batchAct(sharedNetwork, input, [mask], options);
    input.forEach((t) => t.dispose());

    // 3. Enqueue it into the game.
    applyActionToGame(this.tankEid, result.actions);

    // 4. Open the new step.
    if (this.train) {
      this.memory.addFirstPart(state, result.actions, result.logits, result.logProb, mask);
      this.prevPotential = calculateActionReward(this.tankEid);
      this.opened = true;
    }
  }
}
