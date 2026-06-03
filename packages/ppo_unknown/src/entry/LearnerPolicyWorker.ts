import '@tensorflow/tfjs-backend-wasm';
import '../../../ppo/src/infra/unhandledErrors.ts';

import { setConsolePrefix } from '../../../ppo/src/infra/console.ts';
import { initTensorFlow } from '../../../ppo/src/infra/initTensorFlow.ts';
import { metricsChannels } from '../../../ppo/src/infra/channels.ts';
import { episodeSampleChannel } from '../../../ppo/src/core/channels.ts';
import { createLearnerManager } from '../../../ppo/src/learner/createLearnerManager.ts';
import { createPolicyLearnerAgent } from '../../../ppo/src/learner/createPolicyLearnerAgent.ts';
import { CONFIG } from '../config.ts';
import { createInputTensors } from '../state/InputTensors.ts';
import { prepareRandomInputArrays } from '../state/InputArrays.ts';
import { ACTION_HEAD_DIMS, createPolicyNetwork } from '../models/createUnknownNetworks.ts';

setConsolePrefix(`[LEARNER_POLICY]`);

await initTensorFlow('webgpu');

createLearnerManager({ config: CONFIG, createInputTensors, actionHeadDims: ACTION_HEAD_DIMS });

// Forward episode success to the metrics channel for visibility (no curriculum yet).
episodeSampleChannel.obs.subscribe((sample) => {
    metricsChannels.successRatio.postMessage([{
        scenarioIndex: sample.scenarioIndex,
        successRatio: sample.successRatio,
        isReference: sample.isReference,
    }]);
});

createPolicyLearnerAgent({
    config: CONFIG,
    createInputTensors,
    prepareRandomInputArrays,
    actionHeadDims: ACTION_HEAD_DIMS,
    createNetwork: createPolicyNetwork,
});
