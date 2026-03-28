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
import { cos, max, min, sin, sqrt, hypot } from "../../../../../lib/math";
import { clamp } from "lodash";

const AIM_COEFF = 0.0001;
const APPROACH_COEFF = 0.005;
const APPROACH_SATURATION = 5; // BFS cells of sustained approach to reach 3x multiplier
const APPROACH_DECAY = 0.9;
const ENGAGED_RAY_THRESHOLD = 2;
const MOVEMENT_FRAMES = 10;
const MOVEMENT_DIST_THRESHOLD = 200;
const MOVEMENT_REWARD = 0.001;

export function createMlScoreSystem({ world } = GameDI) {
    let allPairsDist: Float32Array | null = null;
    const prevCells = new Map<number, number>();
    const approachAccum = new Map<number, number>();
    const idleRings = new Map<number, RingBuffer<{ x: number, y: number }>>();

    const tick = () => {
        if (!MLState.enabled) return;

        if (allPairsDist == null) {
            const obstacleGrid = computeObstacleGrid(world, GameDI.width, GameDI.height);
            allPairsDist = computeAllPairsDistances(obstacleGrid);
        }

        const vehicleEids = query(world, [Vehicle]);

        for (const vehicleEid of vehicleEids) {
            const playerId = PlayerRef.id[vehicleEid];
            const enemies = findTankEnemiesEids(vehicleEid);
            const enemyRayHits = collectEnemyRayHits(vehicleEid);
            addPathFollowingReward(vehicleEid, playerId, enemies, enemyRayHits);
            addAimReward(vehicleEid, playerId, enemies, enemyRayHits);
            addMovementReward(vehicleEid, playerId);
        }
    };

    const dispose = () => {
        allPairsDist = null;
        prevCells.clear();
        approachAccum.clear();
        idleRings.clear();
    };

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

        // First tick or same cell — no reward
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

        const prevAccum = approachAccum.get(vehicleEid) ?? 0;

        if (totalDelta <= 0) {
            approachAccum.set(vehicleEid, prevAccum * APPROACH_DECAY);
            return;
        }

        const newAccum = prevAccum + totalDelta;
        approachAccum.set(vehicleEid, newAccum);

        // 1x → 3x based on sustained approach distance
        const multiplier = 1 + 2 * min(newAccum / APPROACH_SATURATION, 1);
        Score.addNavigation(playerId, APPROACH_COEFF * multiplier);
    }

    function addMovementReward(vehicleEid: number, playerId: number): void {
        const px = RigidBodyState.position.get(vehicleEid, 0);
        const py = RigidBodyState.position.get(vehicleEid, 1);

        let ring = idleRings.get(vehicleEid);
        if (!ring) {
            ring = new RingBuffer<{ x: number, y: number }>(MOVEMENT_FRAMES);
            idleRings.set(vehicleEid, ring);
        }

        ring.add({ x: px, y: py });

        if (!ring.isFull()) return;

        // Displacement between current and oldest position
        const oldest = ring.getFirst()!;
        const displacement = hypot(px - oldest.x, py - oldest.y);
        const reward = clamp(displacement / MOVEMENT_DIST_THRESHOLD, 0, 1);
        Score.addMovement(playerId, MOVEMENT_REWARD * reward);
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

        if (bestCos <= 0) return;

        Score.addAimAlignment(playerId, bestCos * bestDistFactor * AIM_COEFF);
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