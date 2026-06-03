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
 *   fire [9..14]  — `MASK_NEG` for each direction whose line-of-fire hits a friendly
 *                   Unit before any enemy (no friendly fire), `0` otherwise. Firing
 *                   down an empty line just wastes the shot — the reward, not the
 *                   mask, discourages that.
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
import { OccupantKind } from '../../../unknown/src/Game/Map/HexGrid.ts';
import {
    ACTION_DIM_TOTAL,
    FIRE_DIR_COUNT,
    MASK_NEG,
    MOVE_DIR_COUNT,
    POLICY_ACTION_KIND_COUNT,
} from '../consts.ts';
import { moveDestination } from './applyActionToGame.ts';

const MOVE_OFFSET = POLICY_ACTION_KIND_COUNT; // 3
const FIRE_OFFSET = POLICY_ACTION_KIND_COUNT + MOVE_DIR_COUNT; // 9

export function computeActionMask(eid: number, { world } = GameDI): Float32Array {
    const mask = new Float32Array(ACTION_DIM_TOTAL); // all 0 (kind + fire slices stay 0)
    const grid = MapDI.grid;
    if (!grid) return mask; // no grid → nothing to forbid

    const { RigidBodyState, TeamRef } = getGameComponents(world);

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
    // The projectile travels along the whole direction line, so we walk the ray
    // from the firing cell outward and look at the FIRST blocker it hits:
    //   • friendly Unit first  → forbid (would hit an ally before any enemy);
    //   • enemy Unit / Obstacle first → allow (the round stops on it, no friendly fire);
    //   • Reserved / empty cells are passed through (bullet flies over them).
    // Empty rays stay fireable (0) — a wasted shot is discouraged by the reward,
    // not the mask. The first ray step uses `neighbors(here)[dir]` to match
    // `applyActionToGame`; subsequent steps repeat that axial delta (a straight
    // hex line), which sidesteps the off-grid neighbour-index shift.
    if (here) {
        const myTeam = TeamRef.id[eid];
        const neighbours = grid.neighbors(here);
        let anyFire = false;
        for (let dir = 0; dir < FIRE_DIR_COUNT; dir++) {
            const target = neighbours[dir];
            if (!target) {
                anyFire = true; // off-grid direction → no-op, leave allowed (0)
                continue;
            }
            const dq = target.q - here.q;
            const dr = target.r - here.r;
            let q = target.q;
            let r = target.r;
            let blockedByAlly = false;
            // walk the straight hex line until we leave the grid or hit a blocker
            while (grid.getCell(q, r)) {
                const occupant = grid.getOccupant(q, r);
                if (occupant) {
                    if (occupant.kind === OccupantKind.Unit) {
                        blockedByAlly = TeamRef.id[occupant.eid] === myTeam;
                        break; // first unit hit: ally → forbid, enemy → allow
                    }
                    if (occupant.kind === OccupantKind.Obstacle) break; // bullet stops, no ally hit
                    // OccupantKind.Reserved: physically empty → bullet flies over
                }
                q += dq;
                r += dr;
            }
            if (blockedByAlly) {
                mask[FIRE_OFFSET + dir] = MASK_NEG;
            } else {
                anyFire = true; // 0 = allowed (already)
            }
        }
        // all-masked guard: every direction is blocked by a teammate → leave the slice all 0.
        if (!anyFire) mask.fill(0, FIRE_OFFSET, FIRE_OFFSET + FIRE_DIR_COUNT);
    }

    return mask;
}
