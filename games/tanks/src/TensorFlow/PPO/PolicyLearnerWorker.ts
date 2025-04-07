import '@tensorflow/tfjs-backend-wasm';
import { initTensorFlow } from '../Common/initTensorFlow.ts';
import { setConsolePrefix } from '../Common/console.ts';
import { LearnerManager } from './LearnerManager.ts';
import { PolicyLearnerAgent } from './PolicyLearner/PolicyLearnerAgent.ts';

setConsolePrefix(`[POLICY_LEARNER]`);

// Main initialization function
async function initSystem() {
    // Initialize TensorFlow.js
    const tfInitialized = await initTensorFlow('webgpu');
    if (!tfInitialized) {
        throw new Error('Failed to initialize TensorFlow.js');
    }

    const manager = await LearnerManager.create(new PolicyLearnerAgent());
    manager.start();

    return { manager };
}

initSystem();