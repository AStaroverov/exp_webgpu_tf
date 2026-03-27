import { query } from "bitecs";
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

const AIM_COEFF = 0.0005;
const APPROACH_COEFF = 0.005;
const ENGAGED_RAY_THRESHOLD = 2;

export function createMlScoreSystem({ world } = GameDI) {
    let allPairsDist: Float32Array | null = null;
    const prevCells = new Map<number, number>();

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
        }
    };

    const dispose = () => {
        allPairsDist = null;
        prevCells.clear();
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
        const col = Math.max(0, Math.min(GRID_SIZE - 1, (px / cellW) | 0));
        const row = Math.max(0, Math.min(GRID_SIZE - 1, (py / cellH) | 0));
        const currentCell = row * GRID_SIZE + col;

        const prevCell = prevCells.get(vehicleEid);
        prevCells.set(vehicleEid, currentCell);

        // First tick or same cell — no reward
        if (prevCell === undefined || prevCell === currentCell) return;

        // Enemy already visible in 2+ rays — target found, let combat rewards take over
        for (const count of enemyRayHits.values()) {
            if (count >= ENGAGED_RAY_THRESHOLD) return;
        }

        // Find best enemy to approach (maximizing BFS distance decrease)
        let bestDelta = 0;

        for (let i = 0; i < enemies.length; i++) {
            const eid = enemies[i];
            if (getTankHealth(eid) <= 0) continue;

            const ex = RigidBodyState.position.get(eid, 0);
            const ey = RigidBodyState.position.get(eid, 1);
            const eCol = Math.max(0, Math.min(GRID_SIZE - 1, (ex / cellW) | 0));
            const eRow = Math.max(0, Math.min(GRID_SIZE - 1, (ey / cellH) | 0));
            const enemyCell = eRow * GRID_SIZE + eCol;

            const prevD = dist[prevCell * GRID_CELLS + enemyCell];
            const currD = dist[currentCell * GRID_CELLS + enemyCell];

            if (prevD >= UNREACHABLE || currD >= UNREACHABLE) continue;

            const delta = prevD - currD;
            if (delta > bestDelta) {
                bestDelta = delta;
            }
        }

        if (bestDelta <= 0) return;

        Score.addNavigation(playerId, APPROACH_COEFF);
    }

    function addAimReward(
        vehicleEid: number,
        playerId: number,
        enemies: number[],
        enemyRayHits: Map<number, number>,
    ): void {
        const turretEid = Tank.turretEId[vehicleEid];
        const turretRot = RigidBodyState.rotation[turretEid];

        const fwdX = Math.cos(turretRot);
        const fwdY = Math.sin(turretRot);

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
            const dist = Math.sqrt(dx * dx + dy * dy);
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