/**
 * ppo_unknown consts — action space, decision cadence.
 *
 * Mirrors `ppo_tanks/src/consts.ts` but for the hex-grid, chess-like action space.
 * A PPO "step" here is one DECISION POINT (`needsDecision(eid)`), not one game
 * tick — see PLAN §2 (semi-MDP / options model).
 */

import { POINTY_DIRECTIONS } from "../../unknown/src/Game/Map/HexConfig.ts";
import { MASK_NEG as PPO_MASK_NEG } from "../../ppo/src/core/train.ts";
import { BOARD_CELLS } from "./state/board.ts";

export { FIRE_RING_RADIUS, FIRE_TARGET_OFFSETS, FIRE_CELL_OFFSETS } from "./state/hexNeighbors.ts";

/** Game tick used while simulating headless (ms). Matches tanks' cadence. */
export const TICK_TIME_SIMULATION = Math.round(16 * 1.5);

/** Total experience steps the run targets (curriculum/anneal horizon). */
export const LEARNING_STEPS = 10_000_000;

// ── Action space (single flat categorical) ───────────────────────────────────

/** Move actions: one slot per hex neighbour direction (stable POINTY_DIRECTIONS order). */
export const MOVE_DIR_COUNT = POINTY_DIRECTIONS.length; // 6

/**
 * Fire actions: one slot per BOARD-WINDOW CELL — the policy may fire at ANY
 * reachable cell, not just the rings 1..3 (mechanics: "стрелять в любую
 * достижимую клетку"). A fire action index IS a window cell index, resolved to
 * its hex through `FIRE_CELL_OFFSETS` (state/hexNeighbors.ts). Off-grid /
 * out-of-view / self cells are forbidden by `computeActionMask`, not by a
 * shorter list — the head matches the board the network scores cell-by-cell.
 */
export const FIRE_TARGET_COUNT = BOARD_CELLS; // 121

/**
 * One flat categorical action list (a single policy head samples one index):
 *   [0]            Hold
 *   [1 .. 6]       MoveStep into neighbour direction (index - MOVE_ACTION_OFFSET)
 *   [7 .. 7+CELLS) Fire at window cell `FIRE_CELL_OFFSETS[index - FIRE_ACTION_OFFSET]`
 * A flat space needs no "dead sub-head" handling: a fully masked slice simply
 * can't be sampled, and Hold is never masked, so the distribution stays valid.
 */
export const HOLD_ACTION = 0;
export const MOVE_ACTION_OFFSET = 1;
export const FIRE_ACTION_OFFSET = MOVE_ACTION_OFFSET + MOVE_DIR_COUNT; // 7

/** Total flat action-list / mask length. */
export const ACTION_DIM_TOTAL = 1 + MOVE_DIR_COUNT + FIRE_TARGET_COUNT; // 128

/** Head layout fed to the policy network: a single flat categorical head. */
export const ACTION_HEAD_DIMS = [ACTION_DIM_TOTAL];

/** Additive invalid-action mask sentinel: 0 = allowed, MASK_NEG = forbidden. */
export const MASK_NEG = PPO_MASK_NEG; // -1e9

/**
 * Per-group target shares of the action distribution at init. A single flat
 * softmax over 128 logits hands the 121-wide fire group ~94% of the mass just
 * by count — so early exploration is almost all shooting. These shares rebalance
 * the three groups (hold / move / fire) to deliberate fractions instead.
 */
export const ACTION_GROUP_TARGET_SHARE = { hold: 0.1, move: 0.45, fire: 0.45 } as const;

/**
 * Constant per-action exploration prior (additive logits). With the network's
 * raw logits ≈0 at init, `softmax(ACTION_GROUP_PRIOR)` equals the target shares
 * above, because `prior[i] = log(groupShare / groupSize)`. It is added on top of
 * the validity mask wherever the TRAINED policy turns logits into actions
 * (`computeActionMaskWithPrior`), in BOTH the sample and the train-time
 * logprob/entropy — the same additive constant in old & new logprob keeps the
 * PPO importance ratio consistent. It only shapes the starting point; the
 * network is free to learn logits that move away from it.
 */
export const ACTION_GROUP_PRIOR: Float32Array = (() => {
  const prior = new Float32Array(ACTION_DIM_TOTAL);
  prior[HOLD_ACTION] = Math.log(ACTION_GROUP_TARGET_SHARE.hold / 1);
  for (let d = 0; d < MOVE_DIR_COUNT; d++) {
    prior[MOVE_ACTION_OFFSET + d] = Math.log(ACTION_GROUP_TARGET_SHARE.move / MOVE_DIR_COUNT);
  }
  for (let c = 0; c < FIRE_TARGET_COUNT; c++) {
    prior[FIRE_ACTION_OFFSET + c] = Math.log(ACTION_GROUP_TARGET_SHARE.fire / FIRE_TARGET_COUNT);
  }
  return prior;
})();

// ── Action params ────────────────────────────────────────────────────────────
export const MOVE_SPEED = 1;
export const HOLD_DURATION_MS = 600;
