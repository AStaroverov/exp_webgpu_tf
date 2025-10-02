import '@tensorflow/tfjs-backend-wasm';
import { setConsolePrefix } from '../../../ml-common/console.ts';
import { initTensorFlow } from '../../../ml-common/initTensorFlow.ts';
import '../../../ml-common/uiUtils.ts';
import { createValueLearnerAgent } from './Learner/createValueLearnerAgent.ts';

setConsolePrefix(`[LEARNER_VALUE]`);

await initTensorFlow('wasm');
createValueLearnerAgent();
