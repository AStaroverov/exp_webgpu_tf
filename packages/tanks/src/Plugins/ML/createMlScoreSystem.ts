import { EntityId, query } from "bitecs";
import { Vehicle } from "../../Game/ECS/Components/Vehicle";
import { MLState } from "./MlState";
import { GameDI } from "../../Game/DI/GameDI";
import { Score } from "../../Game/ECS/Components/Score";
import { RAYS_COUNT, RAY_BUFFER, RayHitType, TankInputTensor } from "../Pilots/Components/TankState";
import { PlayerRef } from "../../Game/ECS/Components/PlayerRef";
import { findVehicleFromPart } from "../Pilots/Utils/snapshotTankInputTensor";
import { RigidBodyState } from "../../Game/ECS/Components/Physical";
import { hypot } from "../../../../../lib/math";
import { VehicleController } from "../../Game/ECS/Components/VehicleController";
import { WEIGHTS } from "../../../../ml/src/Reward/calculateReward";

// Track cooldown state per enemy: vehicleEid -> Map<enemyVehicleEid, lostVisibilityTime | -1>
// -1 means on cooldown but still visible, positive number means time when enemy disappeared
const enemyCooldownState = new Map<EntityId, Map<EntityId, number>>();
const ADJACENT_ENEMY_COOLDOWN_MS = 3000;
const COOLDOWN_VISIBLE = -1; // sentinel: on cooldown, enemy still visible

// Track currently visible enemies per vehicle (for exploration reward)
const currentlyVisibleEnemies = new Map<EntityId, Set<EntityId>>();

// Track last reward position per vehicle for exploration reward
const lastRewardPosition = new Map<EntityId, { x: number; y: number }>();

export function createMlScoreSystem({ world } = GameDI) {
    const tick = () => {
        if (!MLState.enabled) return;

        const vehicleEids = query(world, [Vehicle]);
        
        for (const vehicleEid of vehicleEids) {
            const playerId = PlayerRef.id[vehicleEid];
            if (playerId === 0) continue;

            const raysBuffer = TankInputTensor.raysData.getBatch(vehicleEid);
            
            // Adjacent enemy detection reward
            const adjacentEnemyRaysReward = getAdjacentEnemyRaysReward(vehicleEid, raysBuffer);
            if (adjacentEnemyRaysReward > 0) {
                Score.addAdjacentEnemyDetection(playerId, adjacentEnemyRaysReward);
            }
            
            // Exploration reward
            const explorationReward = getExplorationReward(vehicleEid);
            if (explorationReward > 0) {
                Score.addExploration(playerId, explorationReward);
            }
            
            // Proximity penalty
            const proximityPenalty = getProximityPenalty(vehicleEid, raysBuffer);
            if (proximityPenalty > 0) {
                Score.addProximityPenalty(playerId, -proximityPenalty);
            }
        }
    };

    const dispose = () => {
        lastRewardPosition.clear();
        enemyCooldownState.clear();
        currentlyVisibleEnemies.clear();
    };

    return { tick, dispose };
}

export function getAdjacentEnemyRaysReward(vehicleEid: EntityId, raysBuffer: Float64Array): number {
    const now = performance.now();
    
    // Get or create cooldown state map for this vehicle
    let cooldowns = enemyCooldownState.get(vehicleEid);
    if (!cooldowns) {
        cooldowns = new Map();
        enemyCooldownState.set(vehicleEid, cooldowns);
    }

    // Collect adjacent enemy vehicle eids
    const adjacentEnemies = new Set<EntityId>();
    let lastEnemyVehicleEid: EntityId = 0;
    
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
        
        // Check if this enemy is adjacent (same enemy in consecutive rays)
        if (lastEnemyVehicleEid === enemyVehicleEid) {
            adjacentEnemies.add(enemyVehicleEid);
        }
        
        lastEnemyVehicleEid = enemyVehicleEid;
    }

    // Track all visible enemies for exploration reward
    currentlyVisibleEnemies.set(vehicleEid, adjacentEnemies);

    // Update cooldown states for enemies that are no longer visible
    // Cooldown timer only starts when NO enemies are visible at all
    const noEnemiesVisible = adjacentEnemies.size === 0;
    
    for (const [enemyEid, state] of cooldowns) {
        if (!adjacentEnemies.has(enemyEid)) {
            if (state === COOLDOWN_VISIBLE && noEnemiesVisible) {
                // No enemies visible at all - start countdown for this enemy
                cooldowns.set(enemyEid, now);
            } else if (state !== COOLDOWN_VISIBLE && now - state >= ADJACENT_ENEMY_COOLDOWN_MS) {
                // Cooldown expired while enemy not visible - remove
                cooldowns.delete(enemyEid);
            }
        }
    }

    // Check if reward can be given for each visible enemy
    let totalReward = 0;
    for (const enemyEid of adjacentEnemies) {
        const state = cooldowns.get(enemyEid);
        
        if (state === undefined) {
            // No cooldown - give reward and start cooldown
            totalReward += WEIGHTS.ADJACENT_ENEMY_REWARD;
            cooldowns.set(enemyEid, COOLDOWN_VISIBLE);
        } else if (state !== COOLDOWN_VISIBLE && now - state >= ADJACENT_ENEMY_COOLDOWN_MS) {
            // Was invisible, cooldown expired, now visible again - give reward
            totalReward += WEIGHTS.ADJACENT_ENEMY_REWARD;
            cooldowns.set(enemyEid, COOLDOWN_VISIBLE);
        } else if (state !== COOLDOWN_VISIBLE) {
            // Was invisible but cooldown not expired, now visible again - keep on cooldown
            cooldowns.set(enemyEid, COOLDOWN_VISIBLE);
        }
        // else: state === COOLDOWN_VISIBLE - already on cooldown, no reward
    }

    return totalReward;
}

const EXPLORATION_DISTANCE = 50;
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
    
    const hasEnemy = (currentlyVisibleEnemies.get(vehicleEid)?.size ?? 0) > 0;
    return hasEnemy ? WEIGHTS.EXPLORATION_WITH_ENEMY_REWARD : WEIGHTS.EXPLORATION_WITHOUT_ENEMY_REWARD;
}


function getProximityPenalty(vehicleEid: EntityId, raysBuffer: Float64Array): number {
    const move = VehicleController.move[vehicleEid];
    
    // Only penalize when moving
    if (move === 0) return 0;

    const rotation = RigidBodyState.rotation[vehicleEid];
    const colliderRadius = TankInputTensor.colliderRadius[vehicleEid];
    const dangerDistance = colliderRadius;

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
            return WEIGHTS.PROXIMITY_PENALTY;
        }
    }

    return 0;
}
