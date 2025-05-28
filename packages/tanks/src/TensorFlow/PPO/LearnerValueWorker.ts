import '@tensorflow/tfjs-backend-wasm';
import { initTensorFlow } from '../Common/initTensorFlow.ts';
import { setConsolePrefix } from '../Common/console.ts';
import '../Common/uiUtils.ts';
import { createValueLearnerAgent } from './Learner/createValueLearnerAgent.ts';

setConsolePrefix(`[LEARNER_VALUE]`);

await initTensorFlow('wasm');
createValueLearnerAgent();
