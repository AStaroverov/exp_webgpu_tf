import { PI } from '../../../../../lib/math.ts';
import { randomRangeFloat } from '../../../../../lib/random.ts';
import { innerQuery } from 'bitecs';
import { Vehicle, VehicleType } from '../../Game/ECS/Components/Vehicle.ts';
import { getEngine } from './engine.ts';
import { createTank } from '../../Game/ECS/Entities/Tank/createTank.ts';
import { createMeleeCar } from '../../Game/ECS/Entities/MeleeCar/MeleeCar.ts';
import { createHarvester } from '../../Game/ECS/Entities/Harvester/Harvester.ts';
import { RigidBodyState } from '../../Game/ECS/Components/Physical.ts';
import { PlayerEnvDI } from '../../Game/DI/PlayerEnvDI.ts';
import { createPlayer } from '../../Game/ECS/Entities/Player.ts';
import { createRock } from '../../Game/ECS/Entities/Rock/Rock.ts';
// Player team = 0, Enemy team = 1
export const PLAYER_TEAM_ID = 0;
export const ENEMY_TEAM_ID = 1;

// All spawnable vehicle types
export type SpawnableVehicleType = VehicleType;

function createVehicle(type: VehicleType, opts: {
    playerId: number,
    teamId: number,
    x: number,
    y: number,
    rotation: number,
    color: [number, number, number, number],
}) {
    switch (type) {
        case VehicleType.LightTank:
            return createTank({ type: VehicleType.LightTank, ...opts });
        case VehicleType.MediumTank:
            return createTank({ type: VehicleType.MediumTank, ...opts });
        case VehicleType.HeavyTank:
            return createTank({ type: VehicleType.HeavyTank, ...opts });
        case VehicleType.PlayerTank:
            return createTank({ type: VehicleType.PlayerTank, ...opts });
        case VehicleType.MeleeCar:
            return createMeleeCar(opts);
        case VehicleType.Harvester:
            return createHarvester(opts);
        default:
            throw new Error(`Unknown vehicle type ${type}`);
    }
}

export function spawnPlayerTank(vehicleType: VehicleType = VehicleType.LightTank) {
    if (PlayerEnvDI.playerId === null) {
        throw new Error('Player ID is not set');
    }

    const eid = createVehicle(vehicleType, {
        playerId: PlayerEnvDI.playerId,
        teamId: PLAYER_TEAM_ID,
        x: 0,
        y: 0,
        rotation: -PI / 2, // Facing up
        color: [0.2, 0.8, 0.2, 1], // Green player
    });
    
    return eid;
}

export function spawnEnemyAtRandomPosition(vehicleType: VehicleType = VehicleType.LightTank) {
    // Get player position
    let playerX = 0;
    let playerY = 0;
    
    if (PlayerEnvDI.tankEid !== null) {
        playerX = RigidBodyState.position.get(PlayerEnvDI.tankEid, 0);
        playerY = RigidBodyState.position.get(PlayerEnvDI.tankEid, 1);
    }
    
    // Spawn at random distance from player (200-600 units away)
    const spawnDistance = randomRangeFloat(200, 600);
    
    // Random angle
    const angle = randomRangeFloat(0, PI * 2);
    
    const x = playerX + Math.cos(angle) * spawnDistance;
    const y = playerY + Math.sin(angle) * spawnDistance;
    
    // Face towards player
    const rotationToPlayer = Math.atan2(playerY - y, playerX - x);

    return createVehicle(vehicleType, {
        playerId: createPlayer(ENEMY_TEAM_ID),
        teamId: ENEMY_TEAM_ID,
        x,
        y,
        rotation: rotationToPlayer + randomRangeFloat(-PI / 6, PI / 6),
        color: [0.8, 0.2, 0.2, 1], // Red enemies
    });
}

export function getVehicleEids() {
    return innerQuery(getEngine().world, [Vehicle]);
}

export function getEnemyCount() {
    return getVehicleEids().length - 1;
}

/**
 * Spawn a rock at random position near the player
 */
export function spawnRockAtRandomPosition() {
    // Get player position
    let playerX = 0;
    let playerY = 0;
    
    if (PlayerEnvDI.tankEid !== null) {
        playerX = RigidBodyState.position.get(PlayerEnvDI.tankEid, 0);
        playerY = RigidBodyState.position.get(PlayerEnvDI.tankEid, 1);
    }
    
    // Spawn at random distance from player (150-500 units away)
    const spawnDistance = randomRangeFloat(150, 500);
    
    // Random angle
    const angle = randomRangeFloat(0, PI * 2);
    
    const x = playerX + Math.cos(angle) * spawnDistance;
    const y = playerY + Math.sin(angle) * spawnDistance;

    return createRock({ x, y });
}

