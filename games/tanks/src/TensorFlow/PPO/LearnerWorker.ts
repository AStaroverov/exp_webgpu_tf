import '@tensorflow/tfjs-backend-wasm';
import { initTensorFlow } from '../Common/initTensorFlow.ts';
import { setConsolePrefix } from '../Common/console.ts';
import '../Common/utils.ts';
import { LearnerManager } from './Learner/LearnerManager.ts';

setConsolePrefix(`[LEARNER]`);

// Main initialization function
async function initSystem() {
    // Initialize TensorFlow.js
    const tfInitialized = await initTensorFlow('webgpu');
    if (!tfInitialized) {
        throw new Error('Failed to initialize TensorFlow.js');
    }

    const manager = await LearnerManager.create();
    manager.start();

    return { manager };
}

initSystem();