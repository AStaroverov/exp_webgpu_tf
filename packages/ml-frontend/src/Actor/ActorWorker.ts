import '@tensorflow/tfjs-backend-wasm';
import { randomShortId } from '../../../../lib/random.ts';
import { setConsolePrefix } from '../../../ml-common/console.ts';
import { initTensorFlow } from '../../../ml-common/initTensorFlow.ts';
import { EpisodeManager } from './EpisodeManager.ts';

setConsolePrefix(`[ACTOR|${randomShortId()}]`);

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
        (new EpisodeManager()).start();

        console.log('Slave Manager successfully initialized');
    } catch (error) {
        console.error('Failed to start', error);
        return null;
    }
}

initSystem();