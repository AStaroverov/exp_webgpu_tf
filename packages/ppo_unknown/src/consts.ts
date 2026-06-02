/**
 * ppo_unknown consts — action space, decision cadence, scenario sizing.
 *
 * Mirrors `ppo_tanks/src/consts.ts` but for the hex-grid, chess-like action space.
 * A PPO "step" here is one DECISION POINT (`needsDecision(eid)`), not one game
 * tick — see PLAN §2 (semi-MDP / options model).
 */

import { POINTY_DIRECTIONS } from '../../unknown/src/Game/Map/HexConfig.ts';

/** Game tick used while simulating headless (ms). Matches tanks' cadence. */
export const TICK_TIME_SIMULATION = Math.round(16 * 1.5);

/** Total experience steps the run targets (curriculum/anneal horizon). */
export const LEARNING_STEPS = 10_000_000;

// ── Scenario sizing (v1: fixed self-play N-vs-M) ─────────────────────────────
export const TEAMS_COUNT = 2;
export const TEAM_SIZE = 2; // tanks per team

// ── Action space (categorical multi-head) ────────────────────────────────────

/** Discrete action kinds the policy can choose (Aim is folded into Fire for v1). */
export enum PolicyActionKind {
    MoveStep = 0,
    Fire = 1,
    Hold = 2,
}
export const POLICY_ACTION_KIND_COUNT = 3;

/** Move head: one slot per hex neighbour direction (stable POINTY_DIRECTIONS order). */
export const MOVE_DIR_COUNT = POINTY_DIRECTIONS.length; // 6

/** Fire head: fixed-size enemy slot list (k nearest enemies), masked to the chosen one. */
export const K_ENEMY = 3;

/**
 * Head layout fed to the policy network (one categorical head per entry):
 *   [0] kind     — PolicyActionKind
 *   [1] moveDir  — index into the 6 neighbour hexes
 *   [2] fireTgt  — index into the k-nearest-enemy slot list
 */
export const ACTION_HEAD_DIMS = [POLICY_ACTION_KIND_COUNT, MOVE_DIR_COUNT, K_ENEMY];

// ── Action params ────────────────────────────────────────────────────────────
export const MOVE_SPEED = 1;
export const HOLD_DURATION_MS = 600;
