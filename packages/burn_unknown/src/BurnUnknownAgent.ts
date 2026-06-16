/**
 * BurnUnknownAgent — the per-tank actor for the single-thread Burn loop.
 *
 * A faithful re-implementation of ppo_unknown's `UnknownAgent` (env/UnknownAgent.ts)
 * with EXACTLY ONE thing swapped: inference. Where `UnknownAgent.decide()` calls
 * `batchActAsync(tfjsPolicyNetwork, inputTensors, [mask], …)`, this calls
 * `V4Trainer.act(board, mask, greedy)`. Everything else — the two-phase memory API,
 * the reward shaping (`calculateActionReward` × `shapingWeight`), the dead-tank guards,
 * the opening-potential capture timing, the action decode (`applyActionToGame`) — is
 * preserved verbatim so the learning signal matches the real flow.
 *
 * Differences forced by the Burn API:
 *   - `V4Trainer.act` returns `[action, logProb, value]` for a SINGLE observation, so we
 *     feed the flat board directly (no tfjs tensor create/dispose) and additionally
 *     capture the critic VALUE per step. The tfjs flow recomputes values later via
 *     Retrace in the learner; here the value comes from the same forward pass that
 *     sampled the action (epoch-0 ratio == 1), exactly as the V4Trainer.update contract
 *     expects (old_logp/value are behaviour-policy estimates).
 *
 * Reward of a macro-action = Δ(action potential) over its duration × shapingWeight,
 * identical to `UnknownAgent`.
 */

import { AgentMemory, AgentMemoryBatch } from "../../ppo/src/memory/Memory.ts";
import { getTankHealth } from "../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts";
import { prepareInputArrays, InputArrays } from "../../ppo_unknown/src/state/InputArrays.ts";
import { calculateActionReward } from "../../ppo_unknown/src/reward/calculateReward.ts";
import { applyActionToGame } from "../../ppo_unknown/src/env/applyActionToGame.ts";
import { computeActionMask } from "../../ppo_unknown/src/env/computeActionMask.ts";
import { ACTION_DIM_TOTAL } from "../../ppo_unknown/src/consts.ts";
import { ensureTrainer, getGreedyOverride, trainerAct } from "./trainer.ts";

/** A rollout batch flattened for `V4Trainer.update` — parallel arrays, length `size`. */
export type BurnMemoryBatch = AgentMemoryBatch<InputArrays> & {
  /** Behaviour-policy critic value per step (from the act() forward pass). */
  values: Float32Array;
};

export class BurnUnknownAgent {
  private memory = new AgentMemory<InputArrays>();
  // Per-step critic value captured at action time (parallel to memory.states).
  private values: number[] = [];
  private opened = false;
  private prevPotential = 0;

  constructor(
    public readonly tankEid: number,
    public readonly train: boolean,
    public readonly shapingWeight: number,
  ) {}

  getVersion(): number {
    return 0; // overwritten by the manager via the live trainer version
  }

  closeFinalStep(): void {
    if (!this.train || !this.opened) return;
    const delta = (calculateActionReward(this.tankEid) - this.prevPotential) * this.shapingWeight;
    this.memory.updateSecondPart(delta, true);
    this.opened = false;
  }

  getMemoryBatch(finalReward: number): undefined | BurnMemoryBatch {
    if (!this.train) return undefined;
    const batch = this.memory.getBatch(finalReward);
    if (batch == null) return undefined;
    // Trim the parallel value list to the batch's (possibly clamped) length.
    const values = new Float32Array(this.values.slice(0, batch.size));
    return { ...batch, values };
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
    this.values.length = 0;
  }

  /**
   * One decision step — line-for-line the `UnknownAgent.decide()` shape, inference
   * swapped to `V4Trainer.act`. `snapshotUnknownBoard` must already have filled this
   * tank's board row this tick (the driver does it once for all observers).
   */
  async decide(): Promise<void> {
    await ensureTrainer();

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

    // 2. Capture the state AND the opening potential synchronously, then sample
    // asynchronously (matches UnknownAgent's timing: potential read BEFORE this tick's
    // damage is applied, keeping the reward baseline consistent with the closing read).
    const state = prepareInputArrays(this.tankEid);
    const mask = computeActionMask(this.tankEid);
    const nextPotential = calculateActionReward(this.tankEid);
    const greedy = getGreedyOverride() ?? !this.train;

    // V4Trainer.act → Float32Array [action, logProb, value] for the single board.
    const out = await trainerAct(state.board, mask, greedy);
    const action = new Float32Array([out[0]]); // applyActionToGame reads actions[0]
    const logProb = out[1];
    const value = out[2];

    // The tank may have died while inference was in flight.
    if (getTankHealth(this.tankEid) <= 0) {
      this.opened = false;
      return;
    }

    // 3. Enqueue it into the game.
    applyActionToGame(this.tankEid, action);

    // 4. Open the new step. We store the flat mask as the per-step "logits" slot is
    // unused by V4Trainer.update (it needs only action+logProb+value+mask); pass a
    // zero logits buffer to satisfy the AgentMemory shape.
    if (this.train) {
      this.memory.addFirstPart(state, action, ZERO_LOGITS, logProb, mask);
      this.values.push(value);
      this.prevPotential = nextPotential;
      this.opened = true;
    }
  }
}

// AgentMemory requires a logits buffer per step; V4Trainer.update derives everything it
// needs from action/logProb/value/mask, so a shared zero buffer is fine (never read).
const ZERO_LOGITS = new Float32Array(ACTION_DIM_TOTAL);
