/**
 * computeActionMask — build the per-decision invalid-action mask for one agent.
 *
 * The mask is a flat `Float32Array` of length `ACTION_DIM_TOTAL` (15), laid out
 * identically to the concatenated policy logits (kind | move | fire) and additive:
 *   `0`        → action allowed,
 *   `MASK_NEG` → action forbidden (added to the logit before softmax/argmax).
 *
 * Slices (see `ACTION_HEAD_DIMS`):
 *   kind [0..2]   — `MASK_NEG` on `MoveStep` when no neighbour is passable, and on
 *                   `Fire` when every direction is friendly-fire-blocked. `Hold` is
 *                   never masked, so the kind head always has at least one valid
 *                   choice and the policy never *selects* a kind it cannot act on.
 *   move [3..8]   — `0` for each passable hex neighbour (shared predicate with
 *                   `applyActionToGame.moveDestination`), `MASK_NEG` otherwise.
 *   fire [9..14]  — `MASK_NEG` for each direction whose line-of-fire hits a friendly
 *                   Unit before any enemy (no friendly fire), `0` otherwise. Firing
 *                   down an empty line just wastes the shot — the reward, not the
 *                   mask, discourages that.
 *
 * Dead sub-head: when a sub-slice (move/fire) ends up fully `MASK_NEG`, its kind is
 * masked too, so that sub-head is never the acted head. We deliberately leave the
 * slice fully masked rather than resetting it: with `MASK_NEG = -1e9` (not `-inf`)
 * a fully-masked head degenerates to a finite *uniform* distribution — no NaN — and
 * keeping it masked means no spurious gradient flows into a head whose sampled index
 * is meaningless. The act- and train-time masks are identical, so the PPO ratio stays
 * consistent.
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
    PolicyActionKind,
} from '../consts.ts';
import { moveDestination } from './applyActionToGame.ts';

const KIND_OFFSET = 0; // kind slice is first
const MOVE_OFFSET = POLICY_ACTION_KIND_COUNT; // 3
const FIRE_OFFSET = POLICY_ACTION_KIND_COUNT + MOVE_DIR_COUNT; // 9

export function computeActionMask(eid: number, { world } = GameDI): Float32Array {
    const mask = new Float32Array(ACTION_DIM_TOTAL); // all 0 (every action allowed by default)
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
        // No passable neighbour → forbid the MoveStep KIND itself (the move sub-slice
        // stays fully masked → uniform/dead; the policy must pick Fire or Hold instead).
        if (!anyMove) mask[KIND_OFFSET + PolicyActionKind.MoveStep] = MASK_NEG;
    }
    // if `here` is undefined we leave the move slice all 0 (no info → don't forbid).

    // ── fire slice [9..14] ────────────────────────────────────────────────────
    // The projectile travels along the whole direction line, so we walk the ray
    // from the firing cell outward and look at the FIRST blocker it hits:
    //   • friendly Unit first  → forbid (would hit an ally before any enemy);
    //   • enemy Unit / Obstacle first → allow (the round stops on it, no friendly fire);
    //   • Reserved / empty cells are passed through (bullet flies over them).
    // Empty rays stay fireable (0) — a wasted shot is discouraged by the reward,
    // not the mask. The first ray step uses `neighborAt(here, dir)` (direction-stable
    // slot, matching `applyActionToGame`); subsequent steps repeat that axial delta
    // (a straight hex line).
    if (here) {
        const myTeam = TeamRef.id[eid];
        let anyFire = false;
        for (let dir = 0; dir < FIRE_DIR_COUNT; dir++) {
            const target = grid.neighborAt(here, dir);
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
        // Every direction friendly-fire-blocked → forbid the Fire KIND itself (the fire
        // sub-slice stays fully masked → uniform/dead; the policy must pick Move or Hold).
        if (!anyFire) mask[KIND_OFFSET + PolicyActionKind.Fire] = MASK_NEG;
    }

    return mask;
}
