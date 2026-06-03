/**
 * RandomBot — a scripted, non-learning driver for curriculum level 1 ("random").
 *
 * Same decision seam as `UnknownAgent` (the policy driver calls `decide()` when the
 * tank `needsDecision`), but with no network, no memory and no reward bookkeeping:
 * it just steps into a random passable neighbour, so it presents the learner with a
 * MOVING target without any tactics to exploit. When fully boxed in it holds.
 *
 * It reuses `computeActionMask` purely as the source of truth for which neighbour
 * hexes are walkable (shared with the learner), then hands a plain action vector to
 * `applyActionToGame` — exactly the shape `batchAct` would have produced.
 */

import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { MOVE_DIR_COUNT, POLICY_ACTION_KIND_COUNT, PolicyActionKind } from '../consts.ts';
import { applyActionToGame } from './applyActionToGame.ts';
import { computeActionMask } from './computeActionMask.ts';

const MOVE_OFFSET = POLICY_ACTION_KIND_COUNT; // mask move slice starts after the kind slice

export class RandomBot {
    constructor(
        public readonly tankEid: number,
        private readonly di = GameDI,
    ) {}

    decide(): void {
        const mask = computeActionMask(this.tankEid, this.di);

        // Collect the passable neighbour directions (mask 0 = allowed).
        const allowed: number[] = [];
        for (let dir = 0; dir < MOVE_DIR_COUNT; dir++) {
            if (mask[MOVE_OFFSET + dir] === 0) allowed.push(dir);
        }

        const actions = new Float32Array(3);
        if (allowed.length === 0) {
            actions[0] = PolicyActionKind.Hold;
        } else {
            actions[0] = PolicyActionKind.MoveStep;
            actions[1] = allowed[Math.floor(Math.random() * allowed.length)];
        }

        applyActionToGame(this.tankEid, actions, this.di);
    }
}
