/**
 * ppo_unknown consts — action space, decision cadence.
 *
 * Mirrors `ppo_tanks/src/consts.ts` but for the hex-grid, chess-like action space.
 * A PPO "step" here is one DECISION POINT (`needsDecision(eid)`), not one game
 * tick — see PLAN §2 (semi-MDP / options model).
 */

import { POINTY_DIRECTIONS } from "../../unknown/src/Game/Map/HexConfig.ts";
import { MASK_NEG as PPO_MASK_NEG } from "../../ppo/src/core/train.ts";

/** Game tick used while simulating headless (ms). Matches tanks' cadence. */
export const TICK_TIME_SIMULATION = Math.round(16 * 1.5);

/** Total experience steps the run targets (curriculum/anneal horizon). */
export const LEARNING_STEPS = 10_000_000;

// ── Action space (single flat categorical) ───────────────────────────────────

/** Move actions: one slot per hex neighbour direction (stable POINTY_DIRECTIONS order). */
export const MOVE_DIR_COUNT = POINTY_DIRECTIONS.length; // 6

/**
 * Fire actions: one slot per hex neighbour direction (same layout/order as the
 * move slice — stable POINTY_DIRECTIONS). The policy picks one of the 6 nearest
 * hexes to fire at.
 */
export const FIRE_DIR_COUNT = MOVE_DIR_COUNT; // 6

/**
 * One flat categorical action list (a single policy head samples one index):
 *   [0]        Hold
 *   [1 .. 6]   MoveStep into neighbour direction (index - MOVE_ACTION_OFFSET)
 *   [7 .. 12]  Fire at neighbour direction (index - FIRE_ACTION_OFFSET)
 * A flat space needs no "dead sub-head" handling: a fully masked slice simply
 * can't be sampled, and Hold is never masked, so the distribution stays valid.
 */
export const HOLD_ACTION = 0;
export const MOVE_ACTION_OFFSET = 1;
export const FIRE_ACTION_OFFSET = MOVE_ACTION_OFFSET + MOVE_DIR_COUNT; // 7

/** Total flat action-list / mask length. */
export const ACTION_DIM_TOTAL = 1 + MOVE_DIR_COUNT + FIRE_DIR_COUNT; // 13

/** Head layout fed to the policy network: a single flat categorical head. */
export const ACTION_HEAD_DIMS = [ACTION_DIM_TOTAL];

/** Additive invalid-action mask sentinel: 0 = allowed, MASK_NEG = forbidden. */
export const MASK_NEG = PPO_MASK_NEG; // -1e9

// ── Action params ────────────────────────────────────────────────────────────
export const MOVE_SPEED = 1;
export const HOLD_DURATION_MS = 600;
