import { initTensorFlow } from '../../../../../ml-common/initTensorFlow.ts';
import { TankType } from '../../../Game/ECS/Components/Tank.ts';
import { destroyEngine, getEngine } from './engine.ts';
import { addPlayerTank, spawnEnemyOffScreen, BULLET_HELL_MAX_ENEMIES, getEnemyTankEids, resetEngineState } from './engineMethods.ts';
import { toggleBulletHellGame } from './GameState.ts';
import { CurrentActorAgent } from '../../../Pilots/Agents/CurrentActorAgent.ts';
import { getTankHealth } from '../../../Game/ECS/Entities/Tank/TankUtils.ts';
import { randomRangeInt } from '../../../../../../lib/random.ts';

const ENEMY_TANK_TYPES = [TankType.Light, TankType.Medium, TankType.Heavy];

function getRandomTankType(): TankType {
    return ENEMY_TANK_TYPES[randomRangeInt(0, ENEMY_TANK_TYPES.length - 1)];
}

let playerTankEid: number | null = null;

// Setup player tank with keyboard controls
export const setupPlayerTank = (tankType: TankType = TankType.Light) => {
    playerTankEid = addPlayerTank(tankType);
    
    // Enable player controls for this tank
    const engine = getEngine();
    engine.pilots.setPlayerPilot(playerTankEid);
    
    // Enable infinite map mode and set camera to follow player
    engine.setInfiniteMapMode(true);
    engine.setCameraTarget(playerTankEid);
    
    return playerTankEid;
};

// Spawn a single enemy off-screen with random tank type
export const spawnSingleEnemy = async () => {
    const engine = getEngine();
    
    // Check if we reached max enemies
    const currentEnemies = getEnemyTankEids();
    if (currentEnemies.length >= BULLET_HELL_MAX_ENEMIES) {
        return null;
    }
    
    const tankType = getRandomTankType();
    const enemyEid = spawnEnemyOffScreen(tankType);
    
    // Assign AI pilot using TensorFlow model
    const agent = new CurrentActorAgent(enemyEid, false);
    engine.pilots.setPilot(enemyEid, agent);
    
    // Sync pilot if needed (though usually sync is done once at start, 
    // for dynamically spawned agents we might need to ensure model is ready)
    if (agent.sync) {
        await agent.sync();
    }
    
    // Activate the new bot immediately if game is running
    engine.pilots.toggle(true);
    
    return enemyEid;
};

// Finalize game state - sync all pilots
export const finalizeGameState = async () => {
    // Sync all AI agents to load TensorFlow models
    const engine = getEngine();
    const pilots = engine.pilots.getPilots();
    
    await Promise.all(
        pilots.map(pilot => pilot.sync ? pilot.sync() : Promise.resolve())
    );
};

// Activate bot AI
export function activateBots() {
    getEngine().pilots.toggle(true);
}

// Deactivate bot AI
export function deactivateBots() {
    getEngine().pilots.toggle(false);
}

// Start the bullet hell game
export const startBulletHellGame = async (enemyCount: number = 3) => {
    await initTensorFlow('wasm');
    
    // Setup player with special Player tank type
    setupPlayerTank(TankType.Player);
    
    // Spawn initial enemies off-screen with random types
    for (let i = 0; i < enemyCount; i++) {
        await spawnSingleEnemy();
    }
    
    // Finalize and activate
    await finalizeGameState();
    activateBots();
    
    toggleBulletHellGame(true);
};

// Exit the game
export const exitBulletHellGame = () => {
    deactivateBots();
    destroyEngine();
    resetEngineState();
    toggleBulletHellGame(false);
    playerTankEid = null;
};

// Check if player is dead
export const isPlayerDead = (): boolean => {
    if (playerTankEid === null) return false;
    return getTankHealth(playerTankEid) <= 0;
};

// Restart the game
export const restartBulletHellGame = async () => {
    // Cleanup current game
    deactivateBots();
    destroyEngine();
    resetEngineState();
    playerTankEid = null;
    
    // Start new game
    await startBulletHellGame();
};

// Get current player tank
export const getPlayerTankEid = () => playerTankEid;
