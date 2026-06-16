/**
 * ScoreTracker — the ppo_unknown analogue of tanks' `Score` game component, but
 * kept on the training side so the game stays untouched. It maintains TWO cumulative
 * per-player channels, deliberately kept apart because they play different roles:
 *   - COMBAT score (`getScore`) — damage / kills / friendly-fire. The real objective
 *     (a dense proxy of the terminal surviving-health ratio); the agent deltas it as a
 *     real reward that is NEVER annealed.
 *   - APPROACH-SHAPING score (`getShapingScore`) — a heuristic "engage" hint; the agent
 *     deltas it as a potential-based, `shapingWeight`-annealed term.
 * Lumping them would let annealing erase the combat objective, not just the hint.
 *
 * Scored events — combat plus one movement shaping term:
 *   - damage: +DAMAGE_REWARD per point of cross-team damage dealt (the game
 *     attributes damage to the attacker player in `LastHitters`, friendly fire
 *     excluded at source). Damage — not hit-event count — is the unit, so a
 *     12-damage bullet and a stream of 0.05-damage particle overlaps earn the
 *     same reward for the same damage dealt;
 *   - killEnemy: +KILL_REWARD per kill, split between attackers by their damage
 *     share of the dying vehicle (a vehicle tracked last tick and gone now);
 *     every attacker's share is floored at KILL_ASSIST_MIN_REWARD, so even a
 *     tiny assist pays (the shares may then sum past KILL_REWARD);
 *   - friendlyFire: −FRIENDLY_FIRE_PENALTY per point of same-team damage dealt
 *     (from `FriendlyHitters`; self-hits excluded at source, no kill credit);
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
/**
 * Penalty per point of friendly-fire damage dealt (read from `FriendlyHitters`,
 * the same-team twin of `LastHitters`; self-hits are excluded at source).
 * 2 × DAMAGE_REWARD: hurting an ally must cost more than the same damage on an
 * enemy earns, or splash through a teammate would still net positive.
 */
export const FRIENDLY_FIRE_PENALTY = 0.04;
export const KILL_REWARD = 1;
/**
 * Floor for a kill-assist share: anyone who damaged the victim gets at least
 * this much when it dies, however small their damage share (one Medium bullet
 * hit ≈ 0.2 for scale). The shares can then sum past KILL_REWARD — accepted:
 * rewarding every participant matters more than conserving the total.
 */
export const KILL_ASSIST_MIN_REWARD = 0.2;
/**
 * Reward per hex step closer to the nearest enemy (kill = 1, a bullet hit ≈ 0.2
 * for scale). 0.15 while the curriculum is on early rungs: before kills happen,
 * approach is the only dense learning signal and must stay visible next to the
 * rare combat spikes after advantage normalization. Lower it back (~0.05) once
 * combat carries the learning.
 */
export const APPROACH_REWARD = 0.15;

export class ScoreTracker {
  /** playerId → cumulative COMBAT score (damage / kills / friendly-fire). The real
   *  objective signal — read by `calculateActionReward`, never annealed. */
  private score = new Map<number, number>();
  /** playerId → cumulative APPROACH-SHAPING score. A heuristic hint — read by
   *  `calculateShapingPotential` as a potential and faded out by `shapingWeight`. */
  private shapingScore = new Map<number, number>();
  /** victimEid → (attackerPlayerId → last-seen accumulated damage), to diff per tick. */
  private prevDamage = new Map<number, Map<number, number>>();
  /** Same, for friendly-fire damage (`FriendlyHitters`). */
  private prevFriendlyDamage = new Map<number, Map<number, number>>();
  /** vehicleEid → last tick's nearest enemy + hex distance, to diff per tick. */
  private prevApproach = new Map<number, { enemy: number; dist: number }>();

  reset(): void {
    this.score.clear();
    this.shapingScore.clear();
    this.prevDamage.clear();
    this.prevFriendlyDamage.clear();
    this.prevApproach.clear();
  }

  /** Cumulative combat score (the objective). */
  getScore(playerId: number): number {
    return this.score.get(playerId) ?? 0;
  }

  /** Cumulative approach-shaping score (heuristic, consumed as an annealed potential). */
  getShapingScore(playerId: number): number {
    return this.shapingScore.get(playerId) ?? 0;
  }

  private addScore(playerId: number, amount: number): void {
    this.score.set(playerId, (this.score.get(playerId) ?? 0) + amount);
  }

  private addShaping(playerId: number, amount: number): void {
    this.shapingScore.set(playerId, (this.shapingScore.get(playerId) ?? 0) + amount);
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
    const { Vehicle, LastHitters, FriendlyHitters } = getGameComponents(world);
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
        if (inc > 0) this.addScore(playerId, DAMAGE_REWARD * inc);
      });
      this.prevDamage.set(victim, curr);

      // Friendly fire dealt this tick → penalty for the shooter.
      const prevFriendly = this.prevFriendlyDamage.get(victim);
      const currFriendly = new Map<number, number>();
      FriendlyHitters.forEachHitters(victim, (playerId: number, damage: number) => {
        currFriendly.set(playerId, damage);
        const inc = damage - (prevFriendly?.get(playerId) ?? 0);
        if (inc > 0) this.addScore(playerId, -FRIENDLY_FIRE_PENALTY * inc);
      });
      this.prevFriendlyDamage.set(victim, currFriendly);
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
          this.addScore(
            playerId,
            Math.max(KILL_ASSIST_MIN_REWARD, KILL_REWARD * (damage / totalDamage)),
          );
        });
      }
      this.prevDamage.delete(victim);
    }

    for (const victim of this.prevFriendlyDamage.keys()) {
      if (!alive.has(victim)) this.prevFriendlyDamage.delete(victim);
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
        if (TeamRef.id.get(other) === TeamRef.id.get(eid)) continue;
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
        this.addShaping(PlayerRef.id.get(eid), APPROACH_REWARD * (prev.dist - minDist));
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
