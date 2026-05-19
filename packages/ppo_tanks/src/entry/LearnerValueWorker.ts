import '@tensorflow/tfjs-backend-wasm';
import '../../../ppo/src/infra/unhandledErrors.ts';

import { setConsolePrefix } from '../../../ppo/src/infra/console.ts';
import { initTensorFlow } from '../../../ppo/src/infra/initTensorFlow.ts';
import { CONFIG } from '../config.ts';
import { tankStateBindings } from '../state/bindings.ts';
import { createValueNetwork } from '../models/createTankNetworks.ts';
import { createValueLearnerAgent } from '../../../ppo/src/learner/createValueLearnerAgent.ts';

import '../ui/uiUtils.ts';

setConsolePrefix(`[LEARNER_VALUE]`);

await initTensorFlow('webgpu');
createValueLearnerAgent({
    config: CONFIG,
    bindings: tankStateBindings,
    createNetwork: createValueNetwork,
});
