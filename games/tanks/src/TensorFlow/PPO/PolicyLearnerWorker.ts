import '@tensorflow/tfjs-backend-wasm';
import { setBackend } from '../Common/initTensorFlow.ts';
import { setConsolePrefix } from '../Common/console.ts';
import '../Common/uiUtils.ts';
import { LearnerManager } from './LearnerManager.ts';
import { PolicyLearnerAgent } from './PolicyLearner/PolicyLearnerAgent.ts';

setConsolePrefix(`[POLICY_LEARNER]`);

// Main initialization function
async function initSystem() {
    // Initialize TensorFlow.js
    await setBackend('webgpu');

    const manager = await LearnerManager.create(new PolicyLearnerAgent());
    manager.start();

    return { manager };
}

initSystem();