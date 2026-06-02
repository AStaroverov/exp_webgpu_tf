import '@tensorflow/tfjs-backend-wasm';
import '../../../ppo/src/infra/unhandledErrors.ts';

import { setConsolePrefix } from '../../../ppo/src/infra/console.ts';
import { initTensorFlow } from '../../../ppo/src/infra/initTensorFlow.ts';
import { createValueLearnerAgent } from '../../../ppo/src/learner/createValueLearnerAgent.ts';
import { CONFIG } from '../config.ts';
import { unknownStateBindings } from '../state/bindings.ts';
import { createValueNetwork } from '../models/createUnknownNetworks.ts';

setConsolePrefix(`[LEARNER_VALUE]`);

await initTensorFlow('webgpu');

createValueLearnerAgent({
    config: CONFIG,
    bindings: unknownStateBindings,
    createNetwork: createValueNetwork,
});
