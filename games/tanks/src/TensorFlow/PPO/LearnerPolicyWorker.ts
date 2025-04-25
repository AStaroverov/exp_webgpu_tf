import '@tensorflow/tfjs-backend-wasm';
import { initTensorFlow } from '../Common/initTensorFlow.ts';
import { setConsolePrefix } from '../Common/console.ts';
import '../Common/uiUtils.ts';
import { createLearnerManager } from './Learner/createLearnerManager.ts';
import { createPolicyLearnerAgent } from './Learner/createPolicyLearnerAgent.ts';

setConsolePrefix(`[LEARNER_POLICY]`);

// Main initialization function
async function initSystem() {
    // Initialize TensorFlow.js
    const tfInitialized = await initTensorFlow('webgpu');
    if (!tfInitialized) {
        throw new Error('Failed to initialize TensorFlow.js');
    }

    createLearnerManager();
    createPolicyLearnerAgent();
}

initSystem();