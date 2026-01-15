import { initTensorFlow } from '../../../../ml-common/initTensorFlow.ts';
import { VehicleType } from '../../Game/ECS/Components/Vehicle.ts';
import { destroyEngine, getEngine } from './engine.ts';
import { spawnPlayerTank, spawnEnemyAtRandomPosition } from './engineMethods.ts';
import { TestGameState$, toggleTestGame } from './GameState.ts';
import { getTankHealth, syncRemoveTank } from '../../Game/ECS/Entities/Tank/TankUtils.ts';
import { randomRangeInt } from '../../../../../lib/random.ts';
import { PlayerEnvDI } from '../../Game/DI/PlayerEnvDI.ts';
import { Pilot } from '../../Plugins/Pilots/Components/Pilot.ts';

const ENEMY_VEHICLE_TYPES: VehicleType[] = [
    VehicleType.LightTank, 
    VehicleType.MediumTank, 
    VehicleType.HeavyTank,
    VehicleType.MeleeCar,
    VehicleType.Harvester,
];

function getRandomVehicleType(): VehicleType {
    return ENEMY_VEHICLE_TYPES[randomRangeInt(0, ENEMY_VEHICLE_TYPES.length - 1)];
}

// Spawn or replace player vehicle
export function spawnPlayerVehicle(vehicleType: VehicleType) {
    const engine = getEngine();
    
    // Remove old player tank if exists
    if (PlayerEnvDI.tankEid !== null) {
        const oldEid = PlayerEnvDI.tankEid;
        // Dispose agent if exists
        Pilot.disposeAgent(oldEid);
        // Remove tank entity
        syncRemoveTank(oldEid);
        PlayerEnvDI.tankEid = null;
    }
    
    // Create new player tank
    const playerTankEid = spawnPlayerTank(vehicleType);
    engine.setPlayerVehicle(playerTankEid);
    engine.setInfiniteMapMode(true);
    engine.setCameraTarget(playerTankEid);
    
    return playerTankEid;
}

// Spawn a single enemy at random position with specified or random vehicle type
export async function spawnEnemy(vehicleType?: VehicleType) {
    const engine = getEngine();
    
    const type = vehicleType ?? getRandomVehicleType();
    const enemyEid = spawnEnemyAtRandomPosition(type);

    // const agent = new CurrentActorAgent(enemyEid, false);
    // setPilotAgent(enemyEid, agent);
    // engine.pilots.toggle(true);

    // if (agent.sync) { await agent.sync(); }
    
    return enemyEid;
};

// Deactivate bot AI
export function deactivateBots() {
    getEngine().pilots.toggle(false);
}

// Start the test game - no player tank by default, use spawnPlayerVehicle to add one
export async function startTestGame() {
    if (TestGameState$.value.isStarted) return;
    toggleTestGame(true);
    await initTensorFlow('wasm');
    // Initialize engine with infinite map mode
    const engine = getEngine();
    engine.setInfiniteMapMode(true);
}

// Exit the game
export function exitTestGame() {
    deactivateBots();
    destroyEngine();
    toggleTestGame(false);
    PlayerEnvDI.tankEid = null;
    PlayerEnvDI.playerId = null;
};

// Check if player is dead
export function isPlayerDead(): boolean {
    if (PlayerEnvDI.tankEid === null) return false;
    return getTankHealth(PlayerEnvDI.tankEid) <= 0;
};

// Restart the game
export async function restartTestGame() {
    deactivateBots();
    destroyEngine();

    await startTestGame();
};

