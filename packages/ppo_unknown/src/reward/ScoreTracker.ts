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
 *     skipped, not scored, to avoid phantom jumps);
 *   - spot:      per enemy spotted, every contributing unit (proximity / search-
 *     light) earns its ROLE rate × (confidence gain). NOT split between them: the
 *     scout (Ranger) — who cannot deal damage and has no other income — earns the
 *     full RANGER_SPOT_REWARD, while a fighter that merely drove close earns the
 *     smaller FIGHTER_SPOT_REWARD. Attribution is computed once, by the spotting
 *     system, into `Spottable`'s monotonic spotter ledger (the game owns "who spotted
 *     me, how much"); we just diff it per tick, like hits. Confidence jumps to 1 on a
 *     spot and fades over `memoryMs`, so re-spotting a still-lit enemy pays NOTHING
 *     (gain 0); the gain is exactly the faded amount, capping each contributor's
 *     per-enemy income at its role rate per memory window. A self-reveal by firing
 *     (`revealByFire`) has no contributor, hence no ledger entry, hence no points.
 *
 * `update()` must run every tick (hits/deaths happen between decisions); the policy
 * driver calls it. `reset()` is called per episode.
 */

import { query } from 'bitecs';
import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { MapDI } from '../../../unknown/src/Game/DI/MapDI.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';
import { VehicleType } from '../../../unknown/src/Game/ECS/Components/Vehicle.ts';

export const HIT_REWARD = 0.2;
export const KILL_REWARD = 1;
/**
 * Reward per hex step closer to the nearest enemy (kill = 1, hit = 0.2 for
 * scale). 0.15 while the curriculum is on early rungs: before kills happen,
 * approach is the only dense learning signal and must stay visible next to the
 * rare combat spikes after advantage normalization. Lower it back (~0.05) once
 * combat carries the learning.
 */
export const APPROACH_REWARD = 0.15;
/**
 * Reward for a FRESH spot (confidence 0 → 1) of one enemy, per contributing
 * spotter (not split). Scaled by the confidence gain, so each contributor's
 * income per enemy is bounded by the fade rate (at most its role rate per
 * `memoryMs`) — camping or blinking the same target earns no more than letting
 * it fade and re-spotting.
 *
 * The Ranger is a pure scout: no gun, so spotting is its ONLY income — it gets
 * the high rate (above a hit, near a kill). A fighter spotting "with its body"
 * is a side effect of positioning, worth a fraction.
 */
export const RANGER_SPOT_REWARD = 0.4;
export const FIGHTER_SPOT_REWARD = 0.15;

export class ScoreTracker {
    /** playerId → cumulative weighted score. */
    private score = new Map<number, number>();
    /** victimEid → (attackerPlayerId → last-seen hit count), to diff per tick. */
    private prevHits = new Map<number, Map<number, number>>();
    /** vehicleEid → last tick's nearest enemy + hex distance, to diff per tick. */
    private prevApproach = new Map<number, { enemy: number; dist: number }>();
    /** victimEid → (spotterPlayerId → last-seen ledger credit), to diff per tick. */
    private prevSpot = new Map<number, Map<number, number>>();

    reset(): void {
        this.score.clear();
        this.prevHits.clear();
        this.prevApproach.clear();
        this.prevSpot.clear();
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
        this.updateSpotting(world);
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

    /**
     * Spotting reward: diff the per-victim spotter ledger that the spotting system
     * wrote this tick (`Spottable.forEachSpotters`). Each spotter's stored credit is
     * the monotonic sum of the confidence GAINS it caused; the per-tick increase is
     * that spotter's fresh spotting income, scaled by its ROLE rate (Ranger vs
     * fighter — not split, the scout's income must not be diluted by a fighter
     * standing near). The system credits no one for a fire self-reveal, so it scores
     * nothing. Mirrors `updateCombat` — the physics/proximity scan lives once, in the
     * game system, not here.
     */
    private updateSpotting(world = GameDI.world): void {
        const { Vehicle, PlayerRef, Spottable } = getGameComponents(world);
        const vehicles = query(world, [Vehicle, PlayerRef, Spottable]);
        const alive = new Set<number>(vehicles);

        // playerId → role rate, from each potential spotter's vehicle type this tick.
        const rateByPlayer = new Map<number, number>();
        for (let i = 0; i < vehicles.length; i++) {
            const eid = vehicles[i];
            rateByPlayer.set(
                PlayerRef.id[eid],
                Vehicle.type[eid] === VehicleType.Ranger ? RANGER_SPOT_REWARD : FIGHTER_SPOT_REWARD,
            );
        }

        for (let i = 0; i < vehicles.length; i++) {
            const victim = vehicles[i];

            let prev = this.prevSpot.get(victim);
            if (!prev) {
                prev = new Map<number, number>();
                this.prevSpot.set(victim, prev);
            }

            Spottable.forEachSpotters(victim, (playerId: number, credit: number) => {
                const inc = credit - (prev!.get(playerId) ?? 0);
                prev!.set(playerId, credit);
                if (inc > 0) this.add(playerId, (rateByPlayer.get(playerId) ?? 0) * inc);
            });
        }

        for (const victim of this.prevSpot.keys()) {
            if (!alive.has(victim)) this.prevSpot.delete(victim);
        }
    }
}

/** Worker-shared tracker (episodes run sequentially per worker; reset between them). */
export const scoreTracker = new ScoreTracker();
