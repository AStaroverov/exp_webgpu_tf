import '@tensorflow/tfjs-backend-wasm';
import { initTensorFlow } from '../Common/initTensorFlow.ts';
import { createDebugVisualization } from '../Common/debug.ts';
import { setConsolePrefix } from '../Common/console.ts';
import '../Common/uiUtils.ts';
import { CONFIG } from './config.ts';
import { PlayerManager } from './Player/PlayerManager.ts';
import { restoreModels } from '../Models/Trained/restore.ts';
import { macroTasks } from '../../../../../lib/TasksScheduler/macroTasks.ts';

setConsolePrefix(`[TAB]`);

// Main initialization function
async function initSystem() {
    // Initialize TensorFlow.js
    const tfInitialized = await initTensorFlow('wasm');
    if (!tfInitialized) {
        throw new Error('Failed to initialize TensorFlow.js');
    }

    CONFIG.fsModelPath && await restoreModels(CONFIG.fsModelPath);

    Array.from(
        { length: CONFIG.workerCount },
        () => new Worker(new URL('./ActorWorker.ts', import.meta.url), { type: 'module' }),
    );

    macroTasks.addTimeout(() => {
        new Worker(new URL('./LearnerPolicyWorker.ts', import.meta.url), { type: 'module' });
        new Worker(new URL('./LearnerValueWorker.ts', import.meta.url), { type: 'module' });
    }, 1000);

    const playerManager = await PlayerManager.create();
    playerManager.start();

    return { manager: playerManager };
}

initSystem().then(({ manager }) => {
    createDebugVisualization(document.body, manager);
});