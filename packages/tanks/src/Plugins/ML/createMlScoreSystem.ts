import { query } from "bitecs";
import { RingBuffer } from "ring-buffer-ts";
import { Vehicle } from "../../Game/ECS/Components/Vehicle";
import { MLState } from "./MlState";
import { GameDI } from "../../Game/DI/GameDI";
import { RigidBodyState } from "../../Game/ECS/Components/Physical";
import { PlayerRef } from "../../Game/ECS/Components/PlayerRef";
import { Score } from "../../Game/ECS/Components/Score";
import { Tank } from "../../Game/ECS/Components/Tank";
import { getTankHealth } from "../../Game/ECS/Entities/Tank/TankUtils";
import { findTankEnemiesEids, findVehicleFromPart } from "../Pilots/Utils/snapshotTankInputTensor";
import { TankInputTensor, RAY_BUFFER, RAYS_COUNT, RayHitType } from "../Pilots/Components/TankState";
import { computeObstacleGrid } from "../../../../ml-common/computeObstacleGrid";
import { computeAllPairsDistances, UNREACHABLE } from "../../../../ml-common/computeAllPairsDistances";
import { GRID_SIZE, GRID_CELLS } from "../../../../ml/src/Models/Create";
import { SNAPSHOT_EVERY } from "../../../../ml-common/consts";
import { cos, max, min, sin, sqrt, hypot } from "../../../../../lib/math";
import { clamp } from "lodash";

// All coefficients are per-action (1 action = SNAPSHOT_EVERY ticks ≈ 200ms)
const ENGAGED_RAY_THRESHOLD = 2;

const AIM_COEFF = 0.1;

const APPROACH_COEFF = 0.01;

const MOVEMENT_COEFF = 0.05;
const MOVEMENT_ACTIONS = 10; // ~2sec window
const MOVEMENT_DIST_THRESHOLD = 500;

const PROXIMITY_COEFF = 0.005;
const PROXIMITY_RADIUS_MULT = 1.5;

export function createMlScoreSystem({ world } = GameDI) {
    let frame = 0;
    let allPairsDist: Float32Array | null = null;

    const tick = () => {
        if (!MLState.enabled) return;
        return;

        if (allPairsDist == null) {
            const obstacleGrid = computeObstacleGrid(world, GameDI.width, GameDI.height);
            allPairsDist = computeAllPairsDistances(obstacleGrid);
        }

        // Evaluate only once per action (aligned with agent decision step)
        if (frame++ % SNAPSHOT_EVERY !== 0) return;

        const vehicleEids = query(world, [Vehicle]);

        for (const vehicleEid of vehicleEids) {
            const playerId = PlayerRef.id[vehicleEid];
            const enemies = findTankEnemiesEids(vehicleEid);
            const enemyRayHits = collectEnemyRayHits(vehicleEid);
            // addAimReward(vehicleEid, playerId, enemies, enemyRayHits);
            addMovementReward(vehicleEid, playerId);
            addPathFollowingReward(vehicleEid, playerId, enemies, enemyRayHits);
            addProximityPenalty(vehicleEid, playerId);
        }
    };

    const idleRings = new Map<number, RingBuffer<{ x: number, y: number }>>();
    const prevCells = new Map<number, number>();
    // Previous aim score per vehicle (for delta-based reward)
    const prevAimScores = new Map<number, number | null>();
    const prevMinDists = new Map<number, number>();

    const dispose = () => {
        frame = 0;
        allPairsDist = null;
        idleRings.clear();
        prevCells.clear();
        prevAimScores.clear();
        prevMinDists.clear();
    };

    function addMovementReward(vehicleEid: number, playerId: number): void {
        const px = RigidBodyState.position.get(vehicleEid, 0);
        const py = RigidBodyState.position.get(vehicleEid, 1);

        let ring = idleRings.get(vehicleEid);
        if (!ring) {
            ring = new RingBuffer<{ x: number, y: number }>(MOVEMENT_ACTIONS);
            idleRings.set(vehicleEid, ring);
        }

        ring.add({ x: px, y: py });

        if (!ring.isFull()) return;
        const oldest = ring.getFirst();
        if (!oldest) return;
        const displacement = hypot(px - oldest.x, py - oldest.y);
        const reward = clamp(displacement / MOVEMENT_DIST_THRESHOLD, 0, 1);
        Score.addMovement(playerId, MOVEMENT_COEFF * (reward ** 2));
    }

    function addAimReward(
        vehicleEid: number,
        playerId: number,
        enemies: number[],
        enemyRayHits: Map<number, number>,
    ): void {
        const turretEid = Tank.turretEId[vehicleEid];
        const turretRot = RigidBodyState.rotation[turretEid];

        const fwdX = cos(turretRot);
        const fwdY = sin(turretRot);

        const tx = RigidBodyState.position.get(vehicleEid, 0);
        const ty = RigidBodyState.position.get(vehicleEid, 1);

        let bestCos = -1;
        let bestDistFactor = 0;

        for (let i = 0; i < enemies.length; i++) {
            const eid = enemies[i];
            if (getTankHealth(eid) <= 0) continue;
            // Only reward aiming at enemies clearly visible (2+ rays), not through cracks
            if ((enemyRayHits.get(eid) ?? 0) < ENGAGED_RAY_THRESHOLD) continue;

            const ex = RigidBodyState.position.get(eid, 0);
            const ey = RigidBodyState.position.get(eid, 1);
            const dx = ex - tx;
            const dy = ey - ty;
            const dist = sqrt(dx * dx + dy * dy);
            if (dist < 1) continue;

            const cosAngle = (fwdX * dx + fwdY * dy) / dist;
            const distFactor = 1 / (1 + dist / (GameDI.width * 0.5));

            if (cosAngle * distFactor > bestCos * bestDistFactor) {
                bestCos = cosAngle;
                bestDistFactor = distFactor;
            }
        }

        // No visible enemies — reset state to null so next sighting starts fresh
        if (bestCos === -1) {
            prevAimScores.set(vehicleEid, null);
            return;
        }

        const currentScore = bestCos > 0 ? bestCos * bestDistFactor : 0;
        const prevScore = prevAimScores.get(vehicleEid) ?? null;
        prevAimScores.set(vehicleEid, currentScore);

        // First tick after no enemies — skip to establish baseline
        if (prevScore === null) return;

        const delta = currentScore - prevScore;
        if (delta === 0) return;

        const coeff = delta > 0 ? AIM_COEFF : AIM_COEFF * 1.2;
        Score.addAimAlignment(playerId, delta * coeff);
    }

    function addPathFollowingReward(
        vehicleEid: number,
        playerId: number,
        enemies: number[],
        enemyRayHits: Map<number, number>,
    ): void {
        const dist = allPairsDist!;
        const cellW = GameDI.width / GRID_SIZE;
        const cellH = GameDI.height / GRID_SIZE;

        const px = RigidBodyState.position.get(vehicleEid, 0);
        const py = RigidBodyState.position.get(vehicleEid, 1);
        const col = max(0, min(GRID_SIZE - 1, (px / cellW) | 0));
        const row = max(0, min(GRID_SIZE - 1, (py / cellH) | 0));
        const currentCell = row * GRID_SIZE + col;

        const prevCell = prevCells.get(vehicleEid);
        prevCells.set(vehicleEid, currentCell);

        // First action or same cell — no reward
        if (prevCell === undefined || prevCell === currentCell) return;

        // Any enemy engaged (visible in 2+ rays and close) — let combat rewards handle it
        for (const [eid, count] of enemyRayHits) {
            if (count < ENGAGED_RAY_THRESHOLD) continue;
            const ex = RigidBodyState.position.get(eid, 0);
            const ey = RigidBodyState.position.get(eid, 1);
            if (hypot(ex - px, ey - py) < 600) return;
        }

        // Sum approach deltas for all enemies
        let totalDelta = 0;

        for (let i = 0; i < enemies.length; i++) {
            const eid = enemies[i];
            if (getTankHealth(eid) <= 0) continue;

            const ex = RigidBodyState.position.get(eid, 0);
            const ey = RigidBodyState.position.get(eid, 1);
            const eCol = max(0, min(GRID_SIZE - 1, (ex / cellW) | 0));
            const eRow = max(0, min(GRID_SIZE - 1, (ey / cellH) | 0));
            const enemyCell = eRow * GRID_SIZE + eCol;

            const prevD = dist[prevCell * GRID_CELLS + enemyCell];
            const currD = dist[currentCell * GRID_CELLS + enemyCell];

            if (prevD >= UNREACHABLE || currD >= UNREACHABLE) continue;

            const delta = prevD - currD;
            if (delta > 0) {
                totalDelta += delta;
            }
        }

        if (totalDelta <= 0) return;

        Score.addNavigation(playerId, APPROACH_COEFF * totalDelta);
    }

    function addProximityPenalty(vehicleEid: number, playerId: number): void {
        const colliderRadius = TankInputTensor.colliderRadius[vehicleEid];
        if (colliderRadius <= 0) return;

        const threshold = colliderRadius * PROXIMITY_RADIUS_MULT;
        let minDist = Infinity;

        for (let i = 0; i < RAYS_COUNT; i++) {
            const offset = i * RAY_BUFFER;
            const hitType = TankInputTensor.raysData.get(vehicleEid, offset);
            if (hitType === RayHitType.NONE) continue;
            const distance = TankInputTensor.raysData.get(vehicleEid, offset + 6);
            if (distance < minDist) minDist = distance;
        }

        const prev = prevMinDists.get(vehicleEid);
        prevMinDists.set(vehicleEid, minDist);

        // Only penalize when getting closer (distance decreasing) and within threshold
        if (prev === undefined || minDist >= prev || minDist >= threshold) return;

        // violation ∈ (0, 1] — 1 when touching, 0 at threshold
        const violation = (threshold - minDist) / threshold;
        Score.addProximityPenalty(playerId, -PROXIMITY_COEFF * violation);
    }

    return { tick, dispose };
}


function collectEnemyRayHits(vehicleEid: number): Map<number, number> {
    const hits = new Map<number, number>();
    for (let i = 0; i < RAYS_COUNT; i++) {
        const offset = i * RAY_BUFFER;
        if (TankInputTensor.raysData.get(vehicleEid, offset) !== RayHitType.ENEMY_VEHICLE) continue;
        const partEid = TankInputTensor.raysData.get(vehicleEid, offset + 1);
        const ownerEid = findVehicleFromPart(partEid);
        if (ownerEid === 0) continue;
        hits.set(ownerEid, (hits.get(ownerEid) ?? 0) + 1);
    }
    return hits;
}
