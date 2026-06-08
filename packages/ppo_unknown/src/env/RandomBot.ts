/**
 * RandomBot — a scripted, non-learning driver for the curriculum's 'standing' and
 * 'moving' enemy behaviours (both fire sporadically; they differ only in moveProb).
 *
 * Same decision seam as `UnknownAgent` (the policy driver calls `decide()` when the
 * tank `needsDecision`), but with no network, no memory and no reward bookkeeping.
 * Each decision rolls once: with `fireProb` it fires down a random allowed direction,
 * else with `moveProb` it steps into a random passable neighbour, else it holds.
 * So `{ moveProb: 0.3, fireProb: 0 }` is a sometimes-moving target and adding
 * `fireProb` turns it into undirected return fire — pressure without tactics.
 *
 * It reuses `computeActionMask` purely as the source of truth for which actions are
 * legal (shared with the learner), then hands a plain action vector to
 * `applyActionToGame` — exactly the shape `batchAct` would have produced.
 */

import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import {
    FIRE_ACTION_OFFSET,
    FIRE_DIR_COUNT,
    HOLD_ACTION,
    MOVE_ACTION_OFFSET,
    MOVE_DIR_COUNT,
} from '../consts.ts';
import { applyActionToGame } from './applyActionToGame.ts';
import { computeActionMask } from './computeActionMask.ts';

export class RandomBot {
    constructor(
        public readonly tankEid: number,
        private readonly opts: { moveProb: number; fireProb: number },
        private readonly di = GameDI,
    ) {}

    decide(): void {
        const mask = computeActionMask(this.tankEid, this.di);

        const actions = new Float32Array(1);
        actions[0] = this.pickAction(mask);

        applyActionToGame(this.tankEid, actions, this.di);
    }

    private pickAction(mask: Float32Array): number {
        const roll = Math.random();

        if (roll < this.opts.fireProb) {
            const fire = collectAllowed(mask, FIRE_ACTION_OFFSET, FIRE_DIR_COUNT);
            if (fire.length > 0) return fire[Math.floor(Math.random() * fire.length)];
        }

        if (roll < this.opts.fireProb + this.opts.moveProb) {
            const move = collectAllowed(mask, MOVE_ACTION_OFFSET, MOVE_DIR_COUNT);
            if (move.length > 0) return move[Math.floor(Math.random() * move.length)];
        }

        return HOLD_ACTION;
    }
}

/** Collect the allowed action indices in one mask slice (mask 0 = allowed). */
function collectAllowed(mask: Float32Array, offset: number, count: number): number[] {
    const allowed: number[] = [];
    for (let dir = 0; dir < count; dir++) {
        if (mask[offset + dir] === 0) allowed.push(offset + dir);
    }
    return allowed;
}
