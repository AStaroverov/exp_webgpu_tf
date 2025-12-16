import { PI } from '../../../../../lib/math.ts';
import { randomRangeFloat } from '../../../../../lib/random.ts';
import { EntityId, innerQuery } from 'bitecs';
import { Tank, TankType } from '../../Game/ECS/Components/Tank.ts';
import { getEngine } from './engine.ts';
import { createTank } from '../../Game/ECS/Entities/Tank/createTank.ts';
import { createPlayer } from '../../Game/ECS/Entities/Player.ts';
import { RigidBodyState } from '../../Game/ECS/Components/Physical.ts';

export const BULLET_HELL_MAX_ENEMIES = 10;

// Distance from player where enemies spawn
const ENEMY_SPAWN_DISTANCE = 300;

// Player team = 0, Enemy team = 1
export const PLAYER_TEAM_ID = 0;
export const ENEMY_TEAM_ID = 1;

// Store player tank eid for enemy spawn calculations
let playerTankEidForSpawn: EntityId | null = null;
let cachedPlayerId: EntityId | null = null;

function getPlayerId(): EntityId {
    const engine = getEngine();
    if (cachedPlayerId === null) {
        cachedPlayerId = createPlayer(0, engine);
    }
    return cachedPlayerId;
}

export function resetEngineState() {
    playerTankEidForSpawn = null;
    cachedPlayerId = null;
}

export function addPlayerTank(tankType: TankType = TankType.Light) {
    // Player spawns at center of the map (will be camera center)
    const x = 0;
    const y = 0;

    const eid = createTank({
        type: tankType,
        playerId: getPlayerId(),
        teamId: PLAYER_TEAM_ID,
        x,
        y,
        rotation: -PI / 2, // Facing up
        color: [0.2, 0.8, 0.2, 1], // Green player
    });
    
    playerTankEidForSpawn = eid;
    return eid;
}

export function addEnemyTank(slot: number, tankType: TankType = TankType.Light) {
    // Get player position for relative spawn
    let playerX = 0;
    let playerY = 0;
    
    if (playerTankEidForSpawn !== null) {
        playerX = RigidBodyState.position.get(playerTankEidForSpawn, 0);
        playerY = RigidBodyState.position.get(playerTankEidForSpawn, 1);
    }
    
    // Enemies spawn in a semicircle above the player
    const angleSpread = PI * 0.8; // 144 degrees spread
    const startAngle = -PI / 2 - angleSpread / 2; // Start from upper-left
    const angleStep = angleSpread / Math.max(1, BULLET_HELL_MAX_ENEMIES - 1);
    const angle = startAngle + slot * angleStep;
    
    const x = playerX + Math.cos(angle) * ENEMY_SPAWN_DISTANCE;
    const y = playerY + Math.sin(angle) * ENEMY_SPAWN_DISTANCE;
    
    // Face towards player
    const rotationToPlayer = Math.atan2(playerY - y, playerX - x);

    return createTank({
        type: tankType,
        playerId: getPlayerId(),
        teamId: ENEMY_TEAM_ID,
        x,
        y,
        rotation: rotationToPlayer + randomRangeFloat(-PI / 6, PI / 6),
        color: [0.8, 0.2, 0.2, 1], // Red enemies
    });
}

export function spawnEnemyOffScreen(tankType: TankType = TankType.Light) {
    // Get player position
    let playerX = 0;
    let playerY = 0;
    
    if (playerTankEidForSpawn !== null) {
        playerX = RigidBodyState.position.get(playerTankEidForSpawn, 0);
        playerY = RigidBodyState.position.get(playerTankEidForSpawn, 1);
    }
    
    // Calculate spawn distance based on screen size (diagonal + buffer)
    const screenWidth = window.innerWidth;
    const screenHeight = window.innerHeight;
    const spawnDistance = Math.sqrt(screenWidth * screenWidth + screenHeight * screenHeight) / 2 + 100;
    
    // Random angle
    const angle = randomRangeFloat(0, PI * 2);
    
    const x = playerX + Math.cos(angle) * spawnDistance;
    const y = playerY + Math.sin(angle) * spawnDistance;
    
    // Face towards player
    const rotationToPlayer = Math.atan2(playerY - y, playerX - x);

    return createTank({
        type: tankType,
        playerId: getPlayerId(),
        teamId: ENEMY_TEAM_ID,
        x,
        y,
        rotation: rotationToPlayer + randomRangeFloat(-PI / 6, PI / 6),
        color: [0.8, 0.2, 0.2, 1], // Red enemies
    });
}

export function getTankEids() {
    return innerQuery(getEngine().world, [Tank]);
}

export function getPlayerTankEids() {
    const allTanks = Array.from(getTankEids());
    // Filter by team - this would need TeamRef component check
    // For now just return first tank as player
    return allTanks.slice(0, 1);
}

export function getEnemyTankEids() {
    const allTanks = Array.from(getTankEids());
    return allTanks.slice(1);
}

export function getTankType(tankEid: EntityId) {
    return Tank.type[tankEid] as TankType;
}
