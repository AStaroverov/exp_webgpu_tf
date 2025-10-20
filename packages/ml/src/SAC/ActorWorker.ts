import '@tensorflow/tfjs-backend-wasm';
import { randomShortId } from '../../../../lib/random.ts';
import { setConsolePrefix } from '../../../ml-common/console.ts';
import { initTensorFlow } from '../../../ml-common/initTensorFlow.ts';
import '../../../ml-common/unhandledErrors.ts';
import { EpisodeManager } from './Actor/EpisodeManager.ts';

setConsolePrefix(`[SAC-ACTOR|${randomShortId()}]`);

// Main initialization function
async function initSystem() {
    // Initialize TensorFlow.js
    const tfInitialized = await initTensorFlow('wasm');
    if (!tfInitialized) {
        console.error('Failed to initialize TensorFlow.js, aborting');
        return null;
    }

    // Start the episode manager
    try {
        (new EpisodeManager()).start();

        console.log('SAC Actor Worker successfully initialized');
    } catch (error) {
        console.error('Failed to start SAC Actor Worker:', error);
        return null;
    }
}

initSystem();
