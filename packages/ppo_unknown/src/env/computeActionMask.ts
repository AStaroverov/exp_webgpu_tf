/**
 * computeActionMask — build the per-decision invalid-action mask for one agent.
 *
 * The mask is a flat `Float32Array` of length `ACTION_DIM_TOTAL` (73), laid out
 * identically to the concatenated policy logits (kind | move | fire) and additive:
 *   `0`        → action allowed,
 *   `MASK_NEG` → action forbidden (added to the logit before softmax/argmax).
 *
 * Slices (see `ACTION_HEAD_DIMS`):
 *   kind [0..2]   — never masked (an invalid kind+sub-action falls through to a
 *                   no-op in `applyActionToGame`).
 *   move [3..8]   — `0` for each passable hex neighbour (shared predicate with
 *                   `applyActionToGame.moveDestination`), `MASK_NEG` otherwise.
 *   fire [9..14]  — never masked (all `0`): the policy may fire at any of the 6
 *                   neighbour directions. Firing at a hex with no enemy just wastes
 *                   the shot — the reward, not the mask, discourages it.
 *
 * All-masked guard (REFACTOR §2.1): a head with ZERO valid actions must be left
 * fully unmasked (all `0`) — an all-`MASK_NEG` head still yields a (uniform)
 * distribution after softmax but teaches nothing and risks NaN edge cases. Since
 * `kind` is free, the policy can pick `Hold` instead; if it picks the dead kind,
 * `applyActionToGame` no-ops.
 */

import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { MapDI } from '../../../unknown/src/Game/DI/MapDI.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';
import {
    ACTION_DIM_TOTAL,
    MASK_NEG,
    MOVE_DIR_COUNT,
    POLICY_ACTION_KIND_COUNT,
} from '../consts.ts';
import { moveDestination } from './applyActionToGame.ts';

const MOVE_OFFSET = POLICY_ACTION_KIND_COUNT; // 3

export function computeActionMask(eid: number, { world } = GameDI): Float32Array {
    const mask = new Float32Array(ACTION_DIM_TOTAL); // all 0 (kind + fire slices stay 0)
    const grid = MapDI.grid;
    if (!grid) return mask; // no grid → nothing to forbid

    const { RigidBodyState } = getGameComponents(world);

    // ── move slice [3..8] ─────────────────────────────────────────────────────
    const px = RigidBodyState.position.get(eid, 0);
    const py = RigidBodyState.position.get(eid, 1);
    const here = grid.worldToHex(px, py);
    if (here) {
        let anyMove = false;
        for (let dir = 0; dir < MOVE_DIR_COUNT; dir++) {
            if (moveDestination(grid, here.q, here.r, dir)) {
                anyMove = true; // 0 = allowed (already)
            } else {
                mask[MOVE_OFFSET + dir] = MASK_NEG;
            }
        }
        // all-masked guard: fully boxed in → leave the move slice all 0.
        if (!anyMove) mask.fill(0, MOVE_OFFSET, MOVE_OFFSET + MOVE_DIR_COUNT);
    }
    // if `here` is undefined we leave the move slice all 0 (no info → don't forbid).

    // ── fire slice [9..14] ────────────────────────────────────────────────────
    // Left fully unmasked (all 0): any of the 6 neighbour directions is fireable.

    return mask;
}
