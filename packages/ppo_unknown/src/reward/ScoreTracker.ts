/**
 * ScoreTracker — the ppo_unknown analogue of tanks' `Score` game component, but
 * kept on the training side so the game stays untouched. It maintains a cumulative
 * score per player (combat events are monotonic; approach shaping can subtract),
 * exactly the quantity tanks' `calculateActionReward` reads (`Score.getTotalScore`):
 * the agent takes the per-decision DELTA of it.
 *
 * Scored events — combat plus one movement shaping term:
 *   - damage: +DAMAGE_REWARD per point of cross-team damage dealt (the game
 *     attributes damage to the attacker player in `LastHitters`, friendly fire
 *     excluded at source). Damage — not hit-event count — is the unit, so a
 *     12-damage bullet and a stream of 0.05-damage particle overlaps earn the
 *     same reward for the same damage dealt;
 *   - killEnemy: +KILL_REWARD per kill, split between attackers by their damage
 *     share of the dying vehicle (a vehicle tracked last tick and gone now);
 *   - approach:  ±APPROACH_REWARD per hex step closer to / away from the nearest
 *     enemy (per-tick distance delta, so it telescopes over a macro-action; ticks
 *     where the *nearest enemy itself* changed — death or target switch — are
 *     skipped, not scored, to avoid phantom jumps);
 *
 * `update()` must run every tick (hits/deaths happen between decisions); the policy
 * driver calls it. `reset()` is called per episode.
 */

import { query } from "bitecs";
import { GameDI } from "../../../unknown/src/Game/DI/GameDI.ts";
import { MapDI } from "../../../unknown/src/Game/DI/MapDI.ts";
import { getGameComponents } from "../../../unknown/src/Game/ECS/createGameWorld.ts";

/**
 * Reward per point of damage dealt. Scaled so one Medium bullet (12 damage)
 * earns ≈0.2 — the old per-hit reward — while stream weapons, whose damage
 * arrives as hundreds of tiny overlap events, earn proportionally, not per event.
 */
export const DAMAGE_REWARD = 0.02;
export const KILL_REWARD = 1;
/**
 * Reward per hex step closer to the nearest enemy (kill = 1, a bullet hit ≈ 0.2
 * for scale). 0.15 while the curriculum is on early rungs: before kills happen,
 * approach is the only dense learning signal and must stay visible next to the
 * rare combat spikes after advantage normalization. Lower it back (~0.05) once
 * combat carries the learning.
 */
export const APPROACH_REWARD = 0.15;

export class ScoreTracker {
  /** playerId → cumulative weighted score. */
  private score = new Map<number, number>();
  /** victimEid → (attackerPlayerId → last-seen accumulated damage), to diff per tick. */
  private prevDamage = new Map<number, Map<number, number>>();
  /** vehicleEid → last tick's nearest enemy + hex distance, to diff per tick. */
  private prevApproach = new Map<number, { enemy: number; dist: number }>();

  reset(): void {
    this.score.clear();
    this.prevDamage.clear();
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
   * Combat scoring: damage dealt this tick (increase in each victim's per-attacker
   * accumulated damage) and kills (vehicles tracked last tick but gone now → KILL
   * split between attackers by damage share).
   */
  private updateCombat(world = GameDI.world): void {
    const { Vehicle, LastHitters } = getGameComponents(world);
    const vehicles = query(world, [Vehicle, LastHitters]);
    const alive = new Set<number>();

    // Damage dealt this tick = increase in each victim's per-attacker damage.
    for (let i = 0; i < vehicles.length; i++) {
      const victim = vehicles[i];
      alive.add(victim);

      const prev = this.prevDamage.get(victim);
      const curr = new Map<number, number>();
      LastHitters.forEachHitters(victim, (playerId: number, damage: number) => {
        curr.set(playerId, damage);
        const inc = damage - (prev?.get(playerId) ?? 0);
        if (inc > 0) this.add(playerId, DAMAGE_REWARD * inc);
      });
      this.prevDamage.set(victim, curr);
    }

    // Kills = vehicles tracked last tick but gone now → split KILL by damage share.
    for (const [victim, prev] of this.prevDamage) {
      if (alive.has(victim)) continue;

      let totalDamage = 0;
      prev.forEach((damage) => {
        totalDamage += damage;
      });
      if (totalDamage > 0) {
        prev.forEach((damage, playerId) => {
          this.add(playerId, KILL_REWARD * (damage / totalDamage));
        });
      }
      this.prevDamage.delete(victim);
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
      if (!myHex) {
        this.prevApproach.delete(eid);
        continue;
      }

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
        if (dist < minDist) {
          minDist = dist;
          nearest = other;
        }
      }
      if (nearest === 0) {
        this.prevApproach.delete(eid);
        continue;
      }

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
