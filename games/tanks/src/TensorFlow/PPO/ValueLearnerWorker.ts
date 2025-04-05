import '@tensorflow/tfjs-backend-wasm';
import { initTensorFlow } from '../Common/initTensorFlow.ts';
import { setConsolePrefix } from '../Common/console.ts';
import '../Common/uiUtils.ts';
import { LearnerManager } from './LearnerManager.ts';
import { ValueLearnerAgent } from './ValueLearner/ValueLearnerAgent.ts';

setConsolePrefix(`[VALUE_LEARNER]`);

// Main initialization function
async function initSystem() {
    // Initialize TensorFlow.js
    const tfInitialized = await initTensorFlow('wasm');
    if (!tfInitialized) {
        throw new Error('Failed to initialize TensorFlow.js');
    }

    const manager = await LearnerManager.create(new ValueLearnerAgent());
    manager.start();

    return { manager };
}

initSystem();