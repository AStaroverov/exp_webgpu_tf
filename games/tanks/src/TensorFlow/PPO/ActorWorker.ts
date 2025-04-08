import '@tensorflow/tfjs-backend-wasm';
import { initTensorFlow } from '../Common/initTensorFlow.ts';
import { setConsolePrefix } from '../Common/console.ts';
import { getShortRandomId } from '../../../../../lib/random.ts';
import { ActorManager } from './Actor/ActorManager.ts';

setConsolePrefix(`[ACTOR|${ getShortRandomId() }]`);

// Main initialization function
async function initSystem() {
    // Initialize TensorFlow.js
    const tfInitialized = await initTensorFlow('wasm');
    if (!tfInitialized) {
        console.error('Failed to initialize TensorFlow.js, aborting');
        return null;
    }

    // Start the game
    try {
        ActorManager.create().start();

        console.log('Slave Manager successfully initialized');
    } catch (error) {
        console.error('Failed to start', error);
        return null;
    }
}

initSystem();