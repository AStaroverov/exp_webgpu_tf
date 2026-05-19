import GUI from 'lil-gui';
import { VehicleType } from '../../Game/ECS/Components/Vehicle.ts';
import { TestGameState$, TestGameStateEffects, incrementEnemyCount } from '../State/GameState.ts';
import { startTestGame, spawnEnemy, exitTestGame, spawnPlayerVehicle } from '../State/gameMethods.ts';
import { spawnRockAtRandomPosition, spawnBuildingAtRandomPosition, BuildingSize } from '../State/engineMethods.ts';
import { setRenderTarget } from '../State/RenderTarget.ts';
import { upsertModels } from '../../../../ppo/src/models/Restore.ts';
import { initTensorFlow } from '../../../../ppo/src/infra/initTensorFlow.ts';
import { CONFIG } from '../../../../ppo_tanks/src/config.ts';

import '../../../../ppo/src/models/Layers';

const VEHICLE_OPTIONS: Record<string, VehicleType> = {
    '🏎️ Light Tank': VehicleType.LightTank,
    '🛡️ Medium Tank': VehicleType.MediumTank,
    '🦾 Heavy Tank': VehicleType.HeavyTank,
    '⚡ Player Tank': VehicleType.PlayerTank,
    '🚗 Melee Car': VehicleType.MeleeCar,
    '🚜 Harvester': VehicleType.Harvester,
};

const VEHICLE_LABELS = Object.fromEntries(
    Object.entries(VEHICLE_OPTIONS).map(([label, type]) => [type, label])
);

// State object for lil-gui bindings
const state = {
    playerType: VEHICLE_LABELS[VehicleType.PlayerTank],
    enemyType: VEHICLE_LABELS[VehicleType.MediumTank],
    enemyCount: 0,
    
    spawnPlayer() {
        const type = VEHICLE_OPTIONS[this.playerType];
        spawnPlayerVehicle(type);
    },
    
    async spawnEnemy() {
        const type = VEHICLE_OPTIONS[this.enemyType];
        await spawnEnemy(type);
        incrementEnemyCount();
        this.enemyCount = TestGameState$.value.enemyCount;
    },
    
    spawnRock() {
        spawnRockAtRandomPosition();
    },

    buildingSize: 'random' as BuildingSize,
    
    spawnBuilding() {
        spawnBuildingAtRandomPosition(this.buildingSize);
    },
    
    exit() {
        exitTestGame();
        location.reload();
    }
};

// Load persisted values
const savedPlayerType = localStorage.getItem('test-arena-player-type');
const savedEnemyType = localStorage.getItem('test-arena-enemy-type');
if (savedPlayerType && VEHICLE_LABELS[Number(savedPlayerType)]) {
    state.playerType = VEHICLE_LABELS[Number(savedPlayerType)];
}
if (savedEnemyType && VEHICLE_LABELS[Number(savedEnemyType)]) {
    state.enemyType = VEHICLE_LABELS[Number(savedEnemyType)];
}

async function init() {
    await initTensorFlow('wasm');
    await upsertModels(CONFIG.savePath, '/assets/models/v1');
    
    // Setup canvas
    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    setRenderTarget(canvas);
    
    // Start game
    await startTestGame();
    
    // Create GUI
    const gui = new GUI({ title: '🎮 Test Arena' });
    gui.domElement.style.position = 'fixed';
    gui.domElement.style.top = '0';
    gui.domElement.style.right = '0';

    // Click on canvas blurs GUI elements
    canvas.addEventListener('mousedown', () => {
        (document.activeElement as HTMLElement)?.blur();
    });
    
    // Player folder
    const playerFolder = gui.addFolder('👤 Player');
    playerFolder.add(state, 'playerType', Object.keys(VEHICLE_OPTIONS))
        .name('Vehicle')
        .onChange((value: string) => {
            const type = VEHICLE_OPTIONS[value];
            localStorage.setItem('test-arena-player-type', String(type));
        });
    playerFolder.add(state, 'spawnPlayer').name('🚀 Spawn Player');
    
    // Enemy folder
    const enemyFolder = gui.addFolder('👾 Enemy');
    const enemyCountController = enemyFolder.add(state, 'enemyCount')
        .name('Count')
        .disable();
    enemyFolder.add(state, 'enemyType', Object.keys(VEHICLE_OPTIONS))
        .name('Vehicle')
        .onChange((value: string) => {
            const type = VEHICLE_OPTIONS[value];
            localStorage.setItem('test-arena-enemy-type', String(type));
        });
    enemyFolder.add(state, 'spawnEnemy').name('➕ Add Enemy');
    
    // Rock folder
    const rockFolder = gui.addFolder('🪨 Rocks');
    rockFolder.add(state, 'spawnRock').name('➕ Add Rock');

    // Building folder
    const buildingFolder = gui.addFolder('🏚️ Buildings');
    buildingFolder.add(state, 'buildingSize', ['small', 'medium', 'large', 'random'])
        .name('Size');
    buildingFolder.add(state, 'spawnBuilding').name('➕ Add Ruin');
    
    // Subscribe to state updates
    TestGameState$.subscribe(({ enemyCount }) => {
        state.enemyCount = enemyCount;
        enemyCountController.updateDisplay();
    });
    
    // Game controls
    gui.add(state, 'exit').name('🚪 Exit Game');
    
    // Start effects
    TestGameStateEffects().subscribe();
}

init();
