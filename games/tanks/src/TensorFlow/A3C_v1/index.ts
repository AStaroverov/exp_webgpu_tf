import '@tensorflow/tfjs-backend-wasm';
import { MasterManager } from './Master/MasterManager.ts';
import { initTensorFlow } from '../Common/initTensorFlow.ts';
import { createDebugVisualization } from './debug.ts';
import { setConsolePrefix } from '../Common/console.ts';
import '../Common/utils.ts';

setConsolePrefix(`[TAB]`);

// Main initialization function
async function initSystem() {
    console.log('===================================');
    console.log('Initializing System');
    console.log('===================================');

    // Initialize TensorFlow.js
    const tfInitialized = await initTensorFlow();
    if (!tfInitialized) {
        throw new Error('Failed to initialize TensorFlow.js');
    }

    // Start the game
    const [manager] = await Promise.all([
        MasterManager.create(),
        //
        new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }),
        new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }),
        new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }),
        new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }),
        //
        new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }),
        new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }),
        // new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }),
        // new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }),
    ]);

    manager.start();

    console.log('MasterManager successfully initialized');

    return { manager };
}

initSystem().then(({ manager }) => {
    createDebugVisualization(document.body, manager);
});