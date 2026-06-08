/**
 * Spotting system — enemy visibility AND spot-reward attribution, rebuilt every tick.
 *
 * The game is two-sided and a unit is only spotted by the OTHER side, so confidence is
 * a single per-victim value (no observing-team dimension).
 *
 * Runs right after the grid occupancy rebuild and before actions/decisions read
 * the grid, so the snapshot driver (a SystemGroup.Before plugin) sees fresh
 * `visible`/`confidence` the same tick.
 *
 * Phase A — per team, collect the hexes + eids of all its units, and per active
 *           Ranger the vehicles its searchlight lights (a physical capsule sweep,
 *           see Beam.getBeamTargets).
 * Phase B — per unit: decay its confidence, then reinforce it to the strongest active
 *           source among OPPOSING units — beam (lit by an opposing Ranger) outranks
 *           proximity (within `spotRadius` of any opposing unit). `markSpotted` takes
 *           the max, so the value floors at the live source and fades from there once
 *           lost. The per-tick confidence RISE is the spot reward; it is credited
 *           (NEUTRAL, role rates live in training) to every contributing unit via
 *           `Spottable.addSpotCredit`, so the reward layer just diffs the ledger
 *           instead of recomputing the physics here.
 *
 * Module-level scratch structures are reused across ticks (no per-tick alloc, per
 * ECS conventions); they are cleared at the start of each run.
 */

import { query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { MapDI } from '../../../DI/MapDI.ts';
import { getGameComponents } from '../../createGameWorld.ts';
import { SpottingConfig } from '../../../Config/index.ts';
import { VehicleType } from '../../Components/Vehicle.ts';
import { cellKey } from '../../../Map/HexGrid.ts';
import { getBeamTargets, isBeamActive, type BeamCellSink } from '../../Entities/Beam.ts';

interface Axial {
    q: number;
    r: number;
}

/** teamId -> hexes occupied by that team's units this tick (pooled coord objects). */
const teamHexes = new Map<number, Axial[]>();
/** teamId -> eids of that team's units this tick (parallel index to teamHexes). */
const teamUnitEids = new Map<number, number[]>();
/** Reuse coord/eid objects across ticks: per team, how many slots are live this tick. */
const teamHexCount = new Map<number, number>();
/** rangerEid -> eids of vehicles lit by THAT Ranger's searchlight this tick. */
const beamTargetsByRanger = new Map<number, Set<number>>();
/** Reused per-(victim, team) contributor list — pooled, no per-pair alloc. */
const contributorEids: number[] = [];

/**
 * Hex cells covered by ANY Ranger searchlight beam this tick (team-agnostic —
 * light is physically visible to everyone). Rebuilt each tick: `beamCellKeys`
 * dedups (`"q,r"`), the parallel `beamCellQ/R` are the live coords for fast
 * scanning. Read by consumers via `getBeamCells` (e.g. the ppo_unknown snapshot
 * fills its `UnderBeam` board channel from it). Pooled — no per-tick alloc.
 */
const beamCellKeys = new Set<string>();
const beamCellQ: number[] = [];
const beamCellR: number[] = [];
let beamCellCount = 0;

/** Sink passed to `getBeamTargets`: dedup + append into the pooled coord arrays. */
const beamCellSink: BeamCellSink = {
    add(q: number, r: number): void {
        const key = cellKey(q, r);
        if (beamCellKeys.has(key)) return;
        beamCellKeys.add(key);
        if (beamCellCount < beamCellQ.length) {
            beamCellQ[beamCellCount] = q;
            beamCellR[beamCellCount] = r;
        } else {
            beamCellQ.push(q);
            beamCellR.push(r);
        }
        beamCellCount++;
    },
};

/**
 * Per-tick read accessor for the beam-covered hex cells (axial coords). The
 * callback is invoked once per covered cell, bounded to this tick's live count
 * (the pooled arrays may hold stale tail entries from a busier tick). Valid
 * after `updateSpotting` has run for the current tick.
 */
export function getBeamCells(visit: (q: number, r: number) => void): void {
    for (let i = 0; i < beamCellCount; i++) visit(beamCellQ[i], beamCellR[i]);
}

/** True if hex `(q, r)` is covered by any Ranger beam this tick. */
export function isBeamCell(q: number, r: number): boolean {
    return beamCellKeys.has(cellKey(q, r));
}

/** Number of beam-covered cells this tick. */
export function getBeamCellCount(): number {
    return beamCellCount;
}

/** Pure axial hex distance: (|dq| + |dr| + |dq+dr|) / 2. */
function axialDist(aq: number, ar: number, bq: number, br: number): number {
    const dq = aq - bq;
    const dr = ar - br;
    return (Math.abs(dq) + Math.abs(dr) + Math.abs(dq + dr)) / 2;
}

export function createSpottingSystem({ world } = GameDI) {
    const { Vehicle, TeamRef, PlayerRef, RigidBodyState, Spottable } = getGameComponents(world);

    return function updateSpotting(delta: number) {
        const grid = MapDI.grid;
        if (!grid) return;

        // Reset scratch: keep the pooled arrays/sets, just mark them empty.
        for (const teamId of teamHexCount.keys()) teamHexCount.set(teamId, 0);
        for (const set of beamTargetsByRanger.values()) set.clear();
        beamCellKeys.clear();
        beamCellCount = 0;

        const units = query(world, [Vehicle, TeamRef, PlayerRef, RigidBodyState, Spottable]);

        // Phase A — per team: unit hexes + eids, and per Ranger its lit vehicles.
        for (const eid of units) {
            const px = RigidBodyState.position.get(eid, 0);
            const py = RigidBodyState.position.get(eid, 1);
            const hex = grid.worldToHex(px, py);
            if (!hex) continue;

            const teamId = TeamRef.id[eid];

            // Append (hex, eid) into the team's pooled parallel arrays, growing as needed.
            let hexes = teamHexes.get(teamId);
            let eids = teamUnitEids.get(teamId);
            if (!hexes) {
                hexes = [];
                teamHexes.set(teamId, hexes);
            }
            if (!eids) {
                eids = [];
                teamUnitEids.set(teamId, eids);
            }
            const n = teamHexCount.get(teamId) ?? 0;
            if (n < hexes.length) {
                hexes[n].q = hex.q;
                hexes[n].r = hex.r;
                eids[n] = eid;
            } else {
                hexes.push({ q: hex.q, r: hex.r });
                eids.push(eid);
            }
            teamHexCount.set(teamId, n + 1);

            // Only a Ranger whose searchlight is currently pulsing lights anything.
            if (Vehicle.type[eid] === VehicleType.Ranger && isBeamActive(eid)) {
                let set = beamTargetsByRanger.get(eid);
                if (!set) {
                    set = new Set<number>();
                    beamTargetsByRanger.set(eid, set);
                }
                getBeamTargets(eid, set, beamCellSink);
            }
        }

        // Phase B — per unit (victim): gather every opposing-side contributor, then
        // update the unit's single confidence once.
        for (const eid of units) {
            const px = RigidBodyState.position.get(eid, 0);
            const py = RigidBodyState.position.get(eid, 1);
            const hex = grid.worldToHex(px, py);
            if (!hex) continue;

            const ownTeam = TeamRef.id[eid];

            // Enumerate opposing contributors spotting `eid` this tick: a unit within
            // `spotRadius` (proximity) or a Ranger lighting it (beam). The pooled arrays
            // may hold stale tail entries from a busier tick; bound to the live count.
            let contributorCount = 0;
            let near = false;
            let lit = false;
            for (const teamId of teamHexCount.keys()) {
                if (teamId === ownTeam) continue;
                const count = teamHexCount.get(teamId) ?? 0;
                if (count === 0) continue;
                const hexes = teamHexes.get(teamId)!;
                const eids = teamUnitEids.get(teamId)!;
                for (let i = 0; i < count; i++) {
                    const spotter = eids[i];
                    const isNear = axialDist(hex.q, hex.r, hexes[i].q, hexes[i].r) <= SpottingConfig.spotRadius;
                    const isLit = Vehicle.type[spotter] === VehicleType.Ranger
                        && beamTargetsByRanger.get(spotter)?.has(eid) === true;
                    if (!isNear && !isLit) continue;
                    near = near || isNear;
                    lit = lit || isLit;
                    contributorEids[contributorCount++] = spotter;
                }
            }

            // Strongest active source this tick; beam outranks a proximity blip.
            const level = lit
                ? SpottingConfig.confidence.beam
                : near
                  ? SpottingConfig.confidence.proximity
                  : 0;

            // Always decay, then floor at the live source. The per-tick rise is the
            // spot reward; credit it (neutral) to every contributor.
            const before = Spottable.getConfidence(eid);
            Spottable.decay(eid, delta);
            if (level > 0) Spottable.markSpotted(eid, level);
            const gain = Spottable.getConfidence(eid) - before;
            if (gain > 0) {
                for (let i = 0; i < contributorCount; i++) {
                    Spottable.addSpotCredit(eid, PlayerRef.id[contributorEids[i]], gain);
                }
            }
        }
    };
}
