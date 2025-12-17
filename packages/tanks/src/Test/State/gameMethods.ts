import { initTensorFlow } from '../../../../ml-common/initTensorFlow.ts';
import { VehicleType } from '../../Game/ECS/Components/Vehicle.ts';
import { TankVehicleType } from '../../Game/ECS/Entities/Tank/createTank.ts';
import { destroyEngine, getEngine } from './engine.ts';
import { spawnPlayerTank, spawnEnemyAtRandomPosition } from './engineMethods.ts';
import { TestGameState$, toggleTestGame } from './GameState.ts';
import { getTankHealth } from '../../Game/ECS/Entities/Tank/TankUtils.ts';
import { randomRangeInt } from '../../../../../lib/random.ts';
import { PlayerEnvDI } from '../../Game/DI/PlayerEnvDI.ts';
import { CurrentActorAgent } from '../../Pilots/Agents/CurrentActorAgent.ts';

const ENEMY_VEHICLE_TYPES: TankVehicleType[] = [VehicleType.LightTank, VehicleType.MediumTank, VehicleType.HeavyTank];

function getRandomVehicleType(): TankVehicleType {
    return ENEMY_VEHICLE_TYPES[randomRangeInt(0, ENEMY_VEHICLE_TYPES.length - 1)];
}

// Setup player tank with keyboard controls
export function setupPlayerTank(vehicleType: TankVehicleType = VehicleType.PlayerTank) {
    const engine = getEngine();
    const playerTankEid = spawnPlayerTank(vehicleType);
    engine.pilots.setPlayerPilot(playerTankEid);
    engine.setInfiniteMapMode(true);
    engine.setCameraTarget(playerTankEid);
};

// Spawn a single enemy at random position with random tank type
export async function spawnEnemy() {
    const engine = getEngine();
    
    const vehicleType = getRandomVehicleType();
    const enemyEid = spawnEnemyAtRandomPosition(vehicleType);

    const agent = new CurrentActorAgent(enemyEid, false);
    engine.pilots.setPilot(enemyEid, agent);
    engine.pilots.toggle(true);

    if (agent.sync) { await agent.sync(); }
    
    return enemyEid;
};

// Deactivate bot AI
export function deactivateBots() {
    getEngine().pilots.toggle(false);
}

// Start the test game - just player tank, no enemies
export async function startTestGame() {
    if (TestGameState$.value.isStarted) return;
    toggleTestGame(true);
    await initTensorFlow('wasm');
    setupPlayerTank(VehicleType.PlayerTank);
};

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

