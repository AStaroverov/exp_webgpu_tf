import { initTensorFlow } from '../../../../../ml-common/initTensorFlow.ts';
import { TankType } from '../../../Game/ECS/Components/Tank.ts';
import { destroyEngine, getEngine } from './engine.ts';
import { addPlayerTank, addEnemyTank, BULLET_HELL_MAX_ENEMIES } from './engineMethods.ts';
import { toggleBulletHellGame } from './GameState.ts';
import { CurrentActorAgent } from '../../../Pilots/Agents/CurrentActorAgent.ts';

let playerTankEid: number | null = null;

// Setup player tank with keyboard controls
export const setupPlayerTank = (tankType: TankType = TankType.Light) => {
    playerTankEid = addPlayerTank(tankType);
    
    // Enable player controls for this tank
    const engine = getEngine();
    engine.pilots.setPlayerPilot(playerTankEid);
    
    return playerTankEid;
};

// Spawn enemy tanks with AI pilots
export const spawnEnemyWave = async (count: number = 3, tankType: TankType = TankType.Light) => {
    const engine = getEngine();
    const enemies: number[] = [];
    
    for (let i = 0; i < Math.min(count, BULLET_HELL_MAX_ENEMIES); i++) {
        const enemyEid = addEnemyTank(i, tankType);
        enemies.push(enemyEid);
        
        // Assign AI pilot using TensorFlow model
        const agent = new CurrentActorAgent(enemyEid, false);
        engine.pilots.setPilot(enemyEid, agent);
    }
    
    return enemies;
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
    
    // Setup player
    setupPlayerTank(TankType.Light);
    
    // Spawn enemies
    await spawnEnemyWave(enemyCount, TankType.Light);
    
    // Finalize and activate
    await finalizeGameState();
    activateBots();
    
    toggleBulletHellGame(true);
};

// Exit the game
export const exitBulletHellGame = () => {
    deactivateBots();
    destroyEngine();
    toggleBulletHellGame(false);
    playerTankEid = null;
};

// Get current player tank
export const getPlayerTankEid = () => playerTankEid;
