import '@tensorflow/tfjs-backend-wasm';
import '../../../ml-common/unhandledErrors.ts';

import { setConsolePrefix } from '../../../ml-common/console.ts';
import { initTensorFlow } from '../../../ml-common/initTensorFlow.ts';
import { createLearnerManager } from './Learner/createLearnerManager.ts';
import { createPolicyLearnerAgent } from './Learner/createPolicyLearnerAgent.ts';

import '../../../ml-common/uiUtils.ts';

setConsolePrefix(`[LEARNER_POLICY]`);

await initTensorFlow('webgpu');
createLearnerManager();
createPolicyLearnerAgent();