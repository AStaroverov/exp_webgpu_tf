import { initTensorFlow } from '../../../../ml-common/initTensorFlow.ts';
import { VehicleType } from '../../Game/ECS/Components/Vehicle.ts';
import { TankVehicleType } from '../../Game/ECS/Entities/Tank/createTank.ts';
import { destroyEngine, getEngine } from './engine.ts';
import { spawnPlayerTank, spawnEnemyOffScreen, BULLET_HELL_MAX_ENEMIES, getEnemyCount } from './engineMethods.ts';
import { toggleBulletHellGame } from './GameState.ts';
import { getTankHealth } from '../../Game/ECS/Entities/Tank/TankUtils.ts';
import { randomRangeInt } from '../../../../../lib/random.ts';
import { PlayerEnvDI } from '../../Game/DI/PlayerEnvDI.ts';
import { CurrentActorAgent } from '../../Pilots/Agents/CurrentActorAgent.ts';

const ENEMY_VEHICLE_TYPES: TankVehicleType[] = [VehicleType.LightTank, VehicleType.MediumTank, VehicleType.HeavyTank];

function getRandomVehicleType(): TankVehicleType {
    return ENEMY_VEHICLE_TYPES[randomRangeInt(0, ENEMY_VEHICLE_TYPES.length - 1)];
}

// Setup player tank with keyboard controls
export function setupPlayerTank(vehicleType: TankVehicleType = VehicleType.LightTank) {
    const engine = getEngine();
    const playerTankEid = spawnPlayerTank(vehicleType);
    engine.pilots.setPlayerPilot(playerTankEid);
    engine.setInfiniteMapMode(true);
    engine.setCameraTarget(playerTankEid);
};

// Spawn a single enemy off-screen with random tank type
export async function spawnSingleEnemy() {
    const engine = getEngine();
    
    if (getEnemyCount() >= BULLET_HELL_MAX_ENEMIES) {
        return null;
    }
    
    const vehicleType = getRandomVehicleType();
    const enemyEid = spawnEnemyOffScreen(vehicleType);

    const agent = new CurrentActorAgent(enemyEid, false);
    engine.pilots.setPilot(enemyEid, agent);
    engine.pilots.toggle(true)

    if (agent.sync) { await agent.sync(); }
    
    return enemyEid;
};

// Deactivate bot AI
export function deactivateBots() {
    getEngine().pilots.toggle(false);
}

// Start the bullet hell game
export async function startBulletHellGame(enemyCount: number) {
    await initTensorFlow('wasm');
    
    // Setup player with special Player tank type
    setupPlayerTank(VehicleType.PlayerTank);
    
    // Spawn initial enemies off-screen with random types
    for (let i = 0; i < enemyCount; i++) {
        await spawnSingleEnemy();
    }
    
    toggleBulletHellGame(true);
};

// Exit the game
export function exitBulletHellGame() {
    deactivateBots();
    destroyEngine();
    toggleBulletHellGame(false);
    PlayerEnvDI.tankEid = null;
    PlayerEnvDI.playerId = null;
};

// Check if player is dead
export function isPlayerDead(): boolean {
    if (PlayerEnvDI.tankEid === null) return false;
    return getTankHealth(PlayerEnvDI.tankEid) <= 0;
};

// Restart the game
export async function restartBulletHellGame() {
    deactivateBots();
    destroyEngine();

    await startBulletHellGame(3);
};
