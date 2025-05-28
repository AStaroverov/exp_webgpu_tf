import '@tensorflow/tfjs-backend-wasm';
import { initTensorFlow } from '../Common/initTensorFlow.ts';
import { setConsolePrefix } from '../Common/console.ts';
import '../Common/uiUtils.ts';
import { createPolicyLearnerAgent } from './Learner/createPolicyLearnerAgent.ts';
import { createLearnerManager } from './Learner/createLearnerManager.ts';

setConsolePrefix(`[LEARNER_POLICY]`);

await initTensorFlow('webgpu');
createLearnerManager();
createPolicyLearnerAgent();