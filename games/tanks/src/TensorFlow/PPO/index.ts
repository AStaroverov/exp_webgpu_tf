import '@tensorflow/tfjs-backend-wasm';
import { initTensorFlow } from '../Common/initTensorFlow.ts';
import { createDebugVisualization } from '../Common/debug.ts';
import { setConsolePrefix } from '../Common/console.ts';
import '../Common/uiUtils.ts';
import { getCurrentConfig } from './config.ts';
import { PlayerManager } from './Player/PlayerManager.ts';

setConsolePrefix(`[TAB]`);

// Main initialization function
async function initSystem() {
    new Worker(new URL('./PolicyLearnerWorker.ts', import.meta.url), { type: 'module' });
    new Worker(new URL('./ValueLearnerWorker.ts', import.meta.url), { type: 'module' });
    Array.from(
        { length: getCurrentConfig().workerCount },
        () => new Worker(new URL('./ActorWorker.ts', import.meta.url), { type: 'module' }),
    );

    // Initialize TensorFlow.js
    const tfInitialized = await initTensorFlow('wasm');
    if (!tfInitialized) {
        throw new Error('Failed to initialize TensorFlow.js');
    }

    const playerManager = await PlayerManager.create();
    playerManager.start();

    return { manager: playerManager };
}

initSystem().then(({ manager }) => {
    createDebugVisualization(document.body, manager);
});