import { EntityId, query } from "bitecs";
import { Vehicle } from "../Game/ECS/Components/Vehicle";
import { MLState } from "./MlState";
import { GameDI } from "../Game/DI/GameDI";
import { Score } from "../Game/ECS/Components/Score";
import { RAYS_COUNT, RAY_BUFFER, RayHitType, TankInputTensor } from "../Pilots/Components/TankState";
import { PlayerRef } from "../Game/ECS/Components/PlayerRef";
import { findVehicleFromPart } from "../Pilots/Utils/snapshotTankInputTensor";
import { RigidBodyState } from "../Game/ECS/Components/Physical";
import { hypot } from "../../../../lib/math";
import { VehicleController } from "../Game/ECS/Components/VehicleController";

// Track previously detected enemies per vehicle: vehicleEid -> Set<enemyVehicleEid>
const previouslyDetectedEnemies = new Map<EntityId, Set<EntityId>>();

// Track last reward position per vehicle for exploration reward
const EXPLORATION_DISTANCE = 100;
const lastRewardPosition = new Map<EntityId, { x: number; y: number }>();

export function createMlScoreSystem({ world } = GameDI) {
    const tick = () => {
        if (!MLState.enabled) return;

        const vehicleEids = query(world, [Vehicle]);
        
        for (const vehicleEid of vehicleEids) {
            const playerId = PlayerRef.id[vehicleEid];
            if (playerId === 0) continue;

            const raysBuffer = TankInputTensor.raysData.getBatch(vehicleEid);
            const adjacentEnemyRaysReward = getAdjacentEnemyRaysReward(vehicleEid, raysBuffer);
            const explorationReward = getExplorationReward(vehicleEid);
            const proximityPenalty = getProximityPenalty(vehicleEid, raysBuffer);

            const totalReward = (0
                + adjacentEnemyRaysReward
                + explorationReward
                + proximityPenalty
            );
            if (totalReward === 0) continue;
    
            Score.updateScore(playerId, totalReward);
        }
    };

    const dispose = () => {
        lastRewardPosition.clear();
        previouslyDetectedEnemies.clear();
    };

    return { tick, dispose };
}

export function getAdjacentEnemyRaysReward(vehicleEid: EntityId, raysBuffer: Float64Array): number {
    // Get or create the set of previously detected enemies for this vehicle
    let prevDetected = previouslyDetectedEnemies.get(vehicleEid);
    if (!prevDetected) {
        prevDetected = new Set();
        previouslyDetectedEnemies.set(vehicleEid, prevDetected);
    }

    // Collect currently detected enemy vehicle eids
    const currentDetected = new Set<EntityId>();
    let lastEnemyVehicleEid: EntityId = 0;
    let hasNewAdjacentEnemy = false;
    
    // Loop from -1 to RAYS_COUNT to handle wrap-around (last ray -> first ray adjacency)
    for (let i = -1; i < RAYS_COUNT; i++) {
        const idx = (i + RAYS_COUNT) % RAYS_COUNT;
        const hitType = raysBuffer[idx * RAY_BUFFER + 0];
        
        if (hitType !== RayHitType.ENEMY_VEHICLE) {
            lastEnemyVehicleEid = 0;
            continue;
        }
        
        const hitEid = raysBuffer[idx * RAY_BUFFER + 1];
        const enemyVehicleEid = findVehicleFromPart(hitEid);
        
        if (enemyVehicleEid === 0) {
            lastEnemyVehicleEid = 0;
            continue;
        }
        
        const isAdjacent = lastEnemyVehicleEid === enemyVehicleEid;
        const wasDetectedBefore = prevDetected.has(enemyVehicleEid);
        
        if (isAdjacent) {
            if (!wasDetectedBefore) hasNewAdjacentEnemy = true;
            currentDetected.add(enemyVehicleEid);
        } else if (wasDetectedBefore) {
            currentDetected.add(enemyVehicleEid);
        }
        
        lastEnemyVehicleEid = enemyVehicleEid;
    }

    previouslyDetectedEnemies.set(vehicleEid, currentDetected);

    return hasNewAdjacentEnemy ? 2 : 0;
}

function getExplorationReward(vehicleEid: EntityId): number {
    const pos = RigidBodyState.position.getBatch(vehicleEid);
    const x = pos[0];
    const y = pos[1];

    let lastPos = lastRewardPosition.get(vehicleEid);
    if (!lastPos) {
        lastPos = { x, y };
        lastRewardPosition.set(vehicleEid, lastPos);
        return 0;
    }

    const dx = x - lastPos.x;
    const dy = y - lastPos.y;
    const distance = hypot(dx, dy);

    if (distance < EXPLORATION_DISTANCE) {
        return 0;
    }

    lastPos.x = x;
    lastPos.y = y;
    return 0.3;
}


const PROXIMITY_DANGER_MULT = 1;
function getProximityPenalty(vehicleEid: EntityId, raysBuffer: Float64Array): number {
    const move = VehicleController.move[vehicleEid];
    
    // Only penalize when moving
    if (move === 0) return 0;

    const rotation = RigidBodyState.rotation[vehicleEid];
    const colliderRadius = TankInputTensor.colliderRadius[vehicleEid];
    const dangerDistance = PROXIMITY_DANGER_MULT * colliderRadius;

    // Intended movement direction based on rotation and move input
    // move > 0: forward, move < 0: backward
    const forwardX = Math.sin(rotation);
    const forwardY = -Math.cos(rotation);
    const moveX = move > 0 ? forwardX : -forwardX;
    const moveY = move > 0 ? forwardY : -forwardY;

    // Check if intended movement direction points toward any close obstacle
    for (let i = 0; i < RAYS_COUNT; i++) {
        const offset = i * RAY_BUFFER;
        const distance = raysBuffer[offset + 6];

        // Skip rays that are far enough
        if (distance >= dangerDistance || distance === 0) continue;

        // Get ray direction (points from tank to obstacle)
        const dirX = raysBuffer[offset + 4];
        const dirY = raysBuffer[offset + 5];

        // Dot product: positive = moving toward obstacle, negative = moving away
        const dot = dirX * moveX + dirY * moveY;
        
        // Only penalize if intended movement is TOWARD the obstacle
        
        if (dot > 0.5) {
            return -0.1;
        }
    }

    return 0;
}
