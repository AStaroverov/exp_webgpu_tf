import '@tensorflow/tfjs-backend-wasm';
import '../../../ppo/src/infra/unhandledErrors.ts';

import * as tf from '@tensorflow/tfjs';
import { RingBuffer } from 'ring-buffer-ts';
import { setConsolePrefix } from '../../../ppo/src/infra/console.ts';
import { initTensorFlow } from '../../../ppo/src/infra/initTensorFlow.ts';
import { metricsChannels } from '../../../ppo/src/infra/channels.ts';
import { episodeSampleChannel, EpisodeSample } from '../../../ppo/src/core/channels.ts';
import { createLearnerManager } from '../../../ppo/src/learner/createLearnerManager.ts';
import { createPolicyLearnerAgent } from '../../../ppo/src/learner/createPolicyLearnerAgent.ts';
import { CONFIG } from '../config.ts';
import { createInputTensors } from '../state/InputTensors.ts';
import { prepareRandomInputArrays } from '../state/InputArrays.ts';
import { ACTION_HEAD_DIMS, createPolicyNetwork } from '../models/createUnknownNetworks.ts';
import { curriculumStateChannel } from '../curriculumChannel.ts';
import { CurriculumState } from '../curriculum/types.ts';
import { getNetworkCurriculumState, setNetworkCurriculumState } from '../curriculum/curriculumMeta.ts';

setConsolePrefix(`[LEARNER_POLICY]`);

await initTensorFlow('webgpu');

createLearnerManager({ config: CONFIG, createInputTensors, actionHeadDims: ACTION_HEAD_DIMS });

// Per-scenario rolling success window → the curriculum state the actors sample from.
// Mirrors `packages/ppo_tanks/src/entry/LearnerPolicyWorker.ts`.
const mapScenarioIndexToSuccessRatio = new Map<number, RingBuffer<number>>();
const mapScenarioIndexToAvgSuccessRatio = new Map<number, number>();

const computeCurriculumState = (sample: EpisodeSample, prev?: CurriculumState): CurriculumState => {
    if (!mapScenarioIndexToSuccessRatio.has(sample.scenarioIndex)) {
        mapScenarioIndexToSuccessRatio.set(sample.scenarioIndex, new RingBuffer(30));
    }
    mapScenarioIndexToSuccessRatio.get(sample.scenarioIndex)!.add(sample.successRatio);

    for (const [scenarioIndex, successRatioHistory] of mapScenarioIndexToSuccessRatio) {
        const length = successRatioHistory.getBufferLength();
        const size = successRatioHistory.getSize();
        const ratio = length > size / 2
            ? successRatioHistory.toArray().reduce((acc, v) => acc + v, 0) / length
            : prev?.mapScenarioIndexToSuccessRatio[scenarioIndex] ?? 0;

        mapScenarioIndexToAvgSuccessRatio.set(scenarioIndex, ratio);
    }

    return {
        iteration: sample.maxNetworkVersion,
        mapScenarioIndexToSuccessRatio: {
            ...prev?.mapScenarioIndexToSuccessRatio,
            ...Object.fromEntries(mapScenarioIndexToAvgSuccessRatio),
        },
    };
};

function onPolicyReady(network: tf.LayersModel) {
    const lastCurriculumState = getNetworkCurriculumState(network);
    lastCurriculumState && curriculumStateChannel.emit(lastCurriculumState);

    episodeSampleChannel.obs.subscribe((sample) => {
        metricsChannels.successRatio.postMessage([{
            scenarioIndex: sample.scenarioIndex,
            successRatio: sample.successRatio,
            isReference: sample.isReference,
        }]);

        // Only reference (greedy, non-train) episodes update the curriculum, so the
        // gate reflects exploitation skill rather than noisy exploration.
        if (sample.isReference) {
            const curriculumState = computeCurriculumState(sample, getNetworkCurriculumState(network));
            curriculumStateChannel.emit(curriculumState);
            setNetworkCurriculumState(network, curriculumState);
        }
    });
}

createPolicyLearnerAgent({
    config: CONFIG,
    createInputTensors,
    prepareRandomInputArrays,
    actionHeadDims: ACTION_HEAD_DIMS,
    createNetwork: createPolicyNetwork,
    onNetworkReady: onPolicyReady,
});
