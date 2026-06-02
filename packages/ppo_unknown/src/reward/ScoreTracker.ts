/**
 * ScoreTracker — the ppo_unknown analogue of tanks' `Score` game component, but
 * kept on the training side so the game stays untouched. It maintains a MONOTONIC
 * cumulative combat score per player, exactly the quantity tanks' `calculateActionReward`
 * reads (`Score.getTotalScore`): the agent takes the per-decision DELTA of it.
 *
 * Only the simplest combat events count — no physics/aim/speed shaping:
 *   - hitEnemy: +HIT_REWARD per cross-team hit dealt (the game already attributes
 *     hits to the attacker player in `LastHitters`, friendly fire excluded at source);
 *   - killEnemy: +KILL_REWARD per kill, split between attackers by their hit share
 *     of the dying vehicle (a vehicle that was tracked last tick and is gone now).
 *
 * `update()` must run every tick (hits/deaths happen between decisions); the policy
 * driver calls it. `reset()` is called per episode.
 */

import { query } from 'bitecs';
import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';

export const HIT_REWARD = 0.2;
export const KILL_REWARD = 1;

export class ScoreTracker {
    /** playerId → cumulative weighted score. */
    private score = new Map<number, number>();
    /** victimEid → (attackerPlayerId → last-seen hit count), to diff per tick. */
    private prevHits = new Map<number, Map<number, number>>();

    reset(): void {
        this.score.clear();
        this.prevHits.clear();
    }

    getScore(playerId: number): number {
        return this.score.get(playerId) ?? 0;
    }

    private add(playerId: number, amount: number): void {
        this.score.set(playerId, (this.score.get(playerId) ?? 0) + amount);
    }

    update({ world } = GameDI): void {
        const { Vehicle, LastHitters } = getGameComponents(world);
        const vehicles = query(world, [Vehicle, LastHitters]);
        const alive = new Set<number>();

        // Hits dealt this tick = increase in each victim's per-attacker hit count.
        for (let i = 0; i < vehicles.length; i++) {
            const victim = vehicles[i];
            alive.add(victim);

            const prev = this.prevHits.get(victim);
            const curr = new Map<number, number>();
            LastHitters.forEachHitters(victim, (playerId: number, hitCount: number) => {
                curr.set(playerId, hitCount);
                const inc = hitCount - (prev?.get(playerId) ?? 0);
                if (inc > 0) this.add(playerId, HIT_REWARD * inc);
            });
            this.prevHits.set(victim, curr);
        }

        // Kills = vehicles tracked last tick but gone now → split KILL by hit share.
        for (const [victim, prev] of this.prevHits) {
            if (alive.has(victim)) continue;

            let totalHits = 0;
            prev.forEach((count) => { totalHits += count; });
            if (totalHits > 0) {
                prev.forEach((count, playerId) => {
                    this.add(playerId, KILL_REWARD * (count / totalHits));
                });
            }
            this.prevHits.delete(victim);
        }
    }
}

/** Worker-shared tracker (episodes run sequentially per worker; reset between them). */
export const scoreTracker = new ScoreTracker();
