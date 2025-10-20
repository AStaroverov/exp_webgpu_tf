import * as tf from '@tensorflow/tfjs';
import { metricsChannels } from '../../../../ml-common/channels.ts';
import { SAC_CONFIG } from '../../../../ml-common/config.ts';
import { createInputTensors } from '../../../../ml-common/InputTensors.ts';
import { ReplayBuffer } from '../../../../ml-common/ReplayBuffer.ts';
import { syncUnwrapTensor } from '../../../../ml-common/Tensor.ts';
import { getNetworkExpIteration } from '../../../../ml-common/utils.ts';
import { createPolicyNetwork } from '../../Models/Create.ts';
import { Model } from '../../Models/def.ts';
import { getNetwork } from '../../Models/Utils.ts';
import { trainActorNetwork } from '../train.ts';
import { createLearnerAgent } from './createLearnerAgent.ts';
import { LearnData } from './createLearnerManager.ts';
import { isLossDangerous } from './isLossDangerous.ts';

/**
 * SAC Actor (Policy) Learner
 * Trains the actor network to maximize Q(s,a) - α*log π(a|s)
 */
export function createActorLearner() {
    return createLearnerAgent({
        modelName: Model.Policy,
        createNetwork: createPolicyNetwork,
        trainNetwork: trainActor,
    });
}

async function trainActor(network: tf.LayersModel, batch: LearnData) {
    const expIteration = getNetworkExpIteration(network);
    const minLogStd = SAC_CONFIG.minLogStd;
    const maxLogStd = SAC_CONFIG.maxLogStd;
    const alpha = SAC_CONFIG.alpha;
    const batchSize = SAC_CONFIG.miniBatchSize;
    const clipNorm = SAC_CONFIG.clipNorm;

    console.info(`[Train Actor]: Starting...
         Iteration ${expIteration},
         Batch size: ${batch.size},
         Alpha: ${alpha}`);

    // Load critics for actor training
    const [critic1, critic2] = await Promise.all([
        getNetwork(Model.Critic1),
        getNetwork(Model.Critic2),
    ]);

    try {
        const replayBuffer = new ReplayBuffer(batch.states.length);
        const indices = replayBuffer.getSample(batchSize, 0, batch.size);

        // Create mini-batch
        const states = indices.map(i => batch.states[i]);
        const tStates = createInputTensors(states);

        // Train actor
        const actorLoss = trainActorNetwork(
            network,
            critic1,
            critic2,
            tStates,
            alpha,
            batchSize,
            clipNorm,
            minLogStd,
            maxLogStd,
            true, // return cost
        );

        // Log metrics
        if (actorLoss) {
            const actorLossValue = syncUnwrapTensor(actorLoss)[0];
            console.info(`[Train Actor]: Actor Loss = ${actorLossValue.toFixed(4)}`);

            metricsChannels.policyLoss.postMessage([actorLossValue]);

            // Check for dangerous loss
            if (isLossDangerous(actorLossValue)) {
                console.warn('[Train Actor]: Dangerous actor loss detected!');
            }

            actorLoss.dispose();
        }

        // Cleanup tensors
        tStates.forEach(t => t.dispose());

    } finally {
        // Dispose critic networks
        critic1.dispose();
        critic2.dispose();
    }
}
