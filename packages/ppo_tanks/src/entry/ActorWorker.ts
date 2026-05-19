import '@tensorflow/tfjs-backend-wasm';
import { randomShortId } from '../../../../lib/random.ts';
import { setConsolePrefix } from '../../../ppo/src/infra/console.ts';
import { initTensorFlow } from '../../../ppo/src/infra/initTensorFlow.ts';
import '../../../ppo/src/infra/unhandledErrors.ts';
import { TankEpisodeManager } from '../agents/TankEpisodeManager.ts';

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
        (new TankEpisodeManager()).start();

        console.log('Slave Manager successfully initialized');
    } catch (error) {
        console.error('Failed to start', error);
        return null;
    }
}

initSystem();