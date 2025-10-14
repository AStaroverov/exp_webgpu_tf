import '@tensorflow/tfjs-backend-wasm';
import { setConsolePrefix } from '../../../ml-common/console.ts';
import { initTensorFlow } from '../../../ml-common/initTensorFlow.ts';
import { EpisodeManager } from './Actor/EpisodeManager.ts';
import { createLearnerManager } from './Learner/createLearnerManager.ts';
import { createPolicyLearnerAgent } from './Learner/createPolicyLearnerAgent.ts';
import { createValueLearnerAgent } from './Learner/createValueLearnerAgent.ts';

onmessage = (e: MessageEvent) => {
    const idx = Number(e.data);

    if (isNaN(idx)) {
        console.error('Invalid worker index, aborting');
        return;
    }

    setConsolePrefix(`[ACTOR|${idx}]`);
    initSystem(idx);
}

async function initSystem(idx: number) {
    const tfInitialized = await initTensorFlow('wasm');
    if (!tfInitialized) {
        console.error('Failed to initialize TensorFlow.js, aborting');
        return null;
    }

    try {
        createLearnerManager();
        createValueLearnerAgent(idx);
        createPolicyLearnerAgent(idx);

        (new EpisodeManager()).start();

        console.log('Slave Manager successfully initialized');
    } catch (error) {
        console.error('Failed to start', error);
        return null;
    }
}
