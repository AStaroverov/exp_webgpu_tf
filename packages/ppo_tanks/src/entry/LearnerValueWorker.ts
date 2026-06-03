import '@tensorflow/tfjs-backend-wasm';
import '../../../ppo/src/infra/unhandledErrors.ts';

import { setConsolePrefix } from '../../../ppo/src/infra/console.ts';
import { initTensorFlow } from '../../../ppo/src/infra/initTensorFlow.ts';
import { CONFIG } from '../config.ts';
import { createInputTensors } from '../state/InputTensors.ts';
import { prepareRandomInputArrays } from '../state/InputArrays.ts';
import { createValueNetwork } from '../models/createTankNetworks.ts';
import { createValueLearnerAgent } from '../../../ppo/src/learner/createValueLearnerAgent.ts';

import '../ui/uiUtils.ts';

setConsolePrefix(`[LEARNER_VALUE]`);

await initTensorFlow('webgpu');
createValueLearnerAgent({
    config: CONFIG,
    createInputTensors,
    prepareRandomInputArrays,
    createNetwork: createValueNetwork,
});
