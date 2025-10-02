import '@tensorflow/tfjs-backend-wasm';
import { setConsolePrefix } from '../../../ml-common/console.ts';
import { initTensorFlow } from '../../../ml-common/initTensorFlow.ts';
import '../../../ml-common/uiUtils.ts';
import { createLearnerManager } from './Learner/createLearnerManager.ts';
import { createPolicyLearnerAgent } from './Learner/createPolicyLearnerAgent.ts';

setConsolePrefix(`[LEARNER_POLICY]`);

await initTensorFlow('webgpu');
createLearnerManager();
createPolicyLearnerAgent();