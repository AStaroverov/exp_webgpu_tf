import { hypot, PI } from '../../../../../lib/math.ts';
import { randomRangeFloat } from '../../../../../lib/random.ts';
import { innerQuery } from 'bitecs';
import { Vehicle, VehicleType } from '../../Game/ECS/Components/Vehicle.ts';
import { getEngine } from './engine.ts';
import { createTank, TankVehicleType } from '../../Game/ECS/Entities/Tank/createTank.ts';
import { RigidBodyState } from '../../Game/ECS/Components/Physical.ts';
import { PlayerEnvDI } from '../../Game/DI/PlayerEnvDI.ts';
import { createPlayer } from '../../Game/ECS/Entities/Player.ts';

export const BULLET_HELL_MAX_ENEMIES = 10;

// Player team = 0, Enemy team = 1
export const PLAYER_TEAM_ID = 0;
export const ENEMY_TEAM_ID = 1;

export function spawnPlayerTank(vehicleType: TankVehicleType = VehicleType.LightTank) {
    if (PlayerEnvDI.playerId === null) {
        throw new Error('Player ID is not set');
    }

    const eid = createTank({
        type: vehicleType,
        playerId: PlayerEnvDI.playerId,
        teamId: PLAYER_TEAM_ID,
        x: 0,
        y: 0,
        rotation: -PI / 2, // Facing up
        color: [0.2, 0.8, 0.2, 1], // Green player
    });
    
    return eid;
}

export function spawnEnemyOffScreen(vehicleType: TankVehicleType = VehicleType.LightTank) {
    // Get player position
    let playerX = 0;
    let playerY = 0;
    
    if (PlayerEnvDI.tankEid !== null) {
        playerX = RigidBodyState.position.get(PlayerEnvDI.tankEid, 0);
        playerY = RigidBodyState.position.get(PlayerEnvDI.tankEid, 1);
    }
    
    // Calculate spawn distance based on screen size (diagonal + buffer)
    const screenWidth = window.innerWidth;
    const screenHeight = window.innerHeight;
    const spawnDistance = hypot(screenWidth, screenHeight) / 4 + 200;
    
    // Random angle
    const angle = randomRangeFloat(0, PI * 2);
    const x = playerX + Math.cos(angle) * spawnDistance;
    const y = playerY + Math.sin(angle) * spawnDistance;
    
    // Face towards player
    const rotationToPlayer = Math.atan2(playerY - y, playerX - x);

    return createTank({
        type: vehicleType,
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
    return getVehicleEids().length -1;
}
