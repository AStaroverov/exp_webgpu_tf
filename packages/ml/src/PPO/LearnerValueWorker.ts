import '@tensorflow/tfjs-backend-wasm';
import '../../../ml-common/unhandledErrors.ts';

import { setConsolePrefix } from '../../../ml-common/console.ts';
import { initTensorFlow } from '../../../ml-common/initTensorFlow.ts';
import { createValueLearnerAgent } from './Learner/createValueLearnerAgent.ts';

import '../../../ml-common/uiUtils.ts';

setConsolePrefix(`[LEARNER_VALUE]`);

await initTensorFlow('wasm');
createValueLearnerAgent();
