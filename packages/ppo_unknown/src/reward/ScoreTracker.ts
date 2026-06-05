/**
 * ScoreTracker — the ppo_unknown analogue of tanks' `Score` game component, but
 * kept on the training side so the game stays untouched. It maintains a cumulative
 * score per player (combat events are monotonic; approach shaping can subtract),
 * exactly the quantity tanks' `calculateActionReward` reads (`Score.getTotalScore`):
 * the agent takes the per-decision DELTA of it.
 *
 * Scored events — combat plus one movement shaping term:
 *   - hitEnemy: +HIT_REWARD per cross-team hit dealt (the game already attributes
 *     hits to the attacker player in `LastHitters`, friendly fire excluded at source);
 *   - killEnemy: +KILL_REWARD per kill, split between attackers by their hit share
 *     of the dying vehicle (a vehicle that was tracked last tick and is gone now);
 *   - approach:  ±APPROACH_REWARD per hex step closer to / away from the nearest
 *     enemy (per-tick distance delta, so it telescopes over a macro-action; ticks
 *     where the *nearest enemy itself* changed — death or target switch — are
 *     skipped, not scored, to avoid phantom jumps).
 *
 * `update()` must run every tick (hits/deaths happen between decisions); the policy
 * driver calls it. `reset()` is called per episode.
 */

import { query } from 'bitecs';
import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { MapDI } from '../../../unknown/src/Game/DI/MapDI.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';

export const HIT_REWARD = 0.2;
export const KILL_REWARD = 1;
/** Reward per hex step closer to the nearest enemy (kill = 1, hit = 0.2 for scale). */
export const APPROACH_REWARD = 0.05;

export class ScoreTracker {
    /** playerId → cumulative weighted score. */
    private score = new Map<number, number>();
    /** victimEid → (attackerPlayerId → last-seen hit count), to diff per tick. */
    private prevHits = new Map<number, Map<number, number>>();
    /** vehicleEid → last tick's nearest enemy + hex distance, to diff per tick. */
    private prevApproach = new Map<number, { enemy: number; dist: number }>();

    reset(): void {
        this.score.clear();
        this.prevHits.clear();
        this.prevApproach.clear();
    }

    getScore(playerId: number): number {
        return this.score.get(playerId) ?? 0;
    }

    private add(playerId: number, amount: number): void {
        this.score.set(playerId, (this.score.get(playerId) ?? 0) + amount);
    }

    update({ world } = GameDI): void {
        this.updateCombat(world);
        this.updateApproach(world);
    }

    /**
     * Combat scoring: hits dealt this tick (increase in each victim's per-attacker
     * hit count) and kills (vehicles tracked last tick but gone now → KILL split
     * between attackers by hit share).
     */
    private updateCombat(world = GameDI.world): void {
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

    /**
     * Approach shaping: per tick, score the change in hex distance to the nearest
     * enemy (closer → +, away → −). Per-tick deltas telescope, so over a macro-action
     * the agent sees exactly `APPROACH_REWARD × (hexes gained)` — wiggling back and
     * forth nets zero. A tick where the nearest enemy CHANGED (died / overtaken by
     * another) only re-bases the distance, so kills don't leak a phantom penalty.
     */
    private updateApproach(world = GameDI.world): void {
        const grid = MapDI.grid;
        if (!grid) return;

        const { Vehicle, TeamRef, PlayerRef, RigidBodyState } = getGameComponents(world);
        const vehicles = query(world, [Vehicle, TeamRef, PlayerRef, RigidBodyState]);
        const alive = new Set<number>(vehicles);

        for (let i = 0; i < vehicles.length; i++) {
            const eid = vehicles[i];
            const myHex = grid.worldToHex(
                RigidBodyState.position.get(eid, 0),
                RigidBodyState.position.get(eid, 1),
            );
            if (!myHex) { this.prevApproach.delete(eid); continue; }

            // Nearest cross-team vehicle by exact hex distance.
            let nearest = 0;
            let minDist = Infinity;
            for (let j = 0; j < vehicles.length; j++) {
                const other = vehicles[j];
                if (TeamRef.id[other] === TeamRef.id[eid]) continue;
                const otherHex = grid.worldToHex(
                    RigidBodyState.position.get(other, 0),
                    RigidBodyState.position.get(other, 1),
                );
                if (!otherHex) continue;
                const dist = grid.distance(myHex, otherHex);
                if (dist < minDist) { minDist = dist; nearest = other; }
            }
            if (nearest === 0) { this.prevApproach.delete(eid); continue; }

            const prev = this.prevApproach.get(eid);
            if (prev && prev.enemy === nearest) {
                this.add(PlayerRef.id[eid], APPROACH_REWARD * (prev.dist - minDist));
            }
            this.prevApproach.set(eid, { enemy: nearest, dist: minDist });
        }

        for (const eid of this.prevApproach.keys()) {
            if (!alive.has(eid)) this.prevApproach.delete(eid);
        }
    }
}

/** Worker-shared tracker (episodes run sequentially per worker; reset between them). */
export const scoreTracker = new ScoreTracker();
