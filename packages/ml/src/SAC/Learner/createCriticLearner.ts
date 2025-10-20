import * as tf from '@tensorflow/tfjs';
import { metricsChannels } from '../../../../ml-common/channels.ts';
import { SAC_CONFIG } from '../../../../ml-common/config.ts';
import { createInputTensors } from '../../../../ml-common/InputTensors.ts';
import { ReplayBuffer } from '../../../../ml-common/ReplayBuffer.ts';
import { syncUnwrapTensor } from '../../../../ml-common/Tensor.ts';
import { getNetworkExpIteration } from '../../../../ml-common/utils.ts';
import { createCriticNetwork } from '../../Models/Create.ts';
import { Model } from '../../Models/def.ts';
import { getNetwork, softUpdateTargetNetwork } from '../../Models/Utils.ts';
import { trainCriticNetworks } from '../train.ts';
import { createLearnerAgent } from './createLearnerAgent.ts';
import { LearnData } from './createLearnerManager.ts';
import { isLossDangerous } from './isLossDangerous.ts';

/**
 * SAC Critic Learner
 * Trains both critic networks (Q1 and Q2) and updates target networks
 */
export function createCritic1Learner() {
    return createLearnerAgent({
        modelName: Model.Critic1,
        createNetwork: () => createCriticNetwork(Model.Critic1),
        trainNetwork: trainCritics,
    });
}

export function createCritic2Learner() {
    return createLearnerAgent({
        modelName: Model.Critic2,
        createNetwork: () => createCriticNetwork(Model.Critic2),
        trainNetwork: trainCritics,
    });
}

async function trainCritics(network: tf.LayersModel, batch: LearnData) {
    const expIteration = getNetworkExpIteration(network);
    const minLogStd = SAC_CONFIG.minLogStd;
    const maxLogStd = SAC_CONFIG.maxLogStd;
    const alpha = SAC_CONFIG.alpha;
    const gamma = SAC_CONFIG.gamma;
    const tau = SAC_CONFIG.tau;
    const batchSize = SAC_CONFIG.miniBatchSize;
    const clipNorm = SAC_CONFIG.clipNorm;

    console.info(`[Train Critics]: Starting...
         Iteration ${expIteration},
         Batch size: ${batch.size},
         Gamma: ${gamma}, Tau: ${tau}`);

    // Load all required networks
    const [critic1, critic2, targetCritic1, targetCritic2, actor] = await Promise.all([
        getNetwork(Model.Critic1),
        getNetwork(Model.Critic2),
        getNetwork(Model.TargetCritic1, () => createCriticNetwork(Model.TargetCritic1)),
        getNetwork(Model.TargetCritic2, () => createCriticNetwork(Model.TargetCritic2)),
        getNetwork(Model.Policy),
    ]);

    try {
        const replayBuffer = new ReplayBuffer(batch.states.length);
        const indices = replayBuffer.getSample(batchSize, 0, batch.size);

        // Create mini-batch
        const states = indices.map(i => batch.states[i]);
        const nextStates = indices.map(i => batch.nextStates[i]);
        const actions = indices.map(i => batch.actions[i]);
        const rewards = indices.map(i => batch.rewards[i]);
        const dones = indices.map(i => batch.dones[i]);

        const tStates = createInputTensors(states);
        const tNextStates = createInputTensors(nextStates);
        const tActions = tf.tensor2d(actions.map(a => Array.from(a)));
        const tRewards = tf.tensor1d(rewards);
        const tDones = tf.tensor1d(dones);

        // Train critics
        const { loss1, loss2 } = trainCriticNetworks(
            critic1,
            critic2,
            targetCritic1,
            targetCritic2,
            actor,
            {
                states: tStates,
                actions: tActions,
                rewards: tRewards,
                nextStates: tNextStates,
                dones: tDones,
            },
            alpha,
            gamma,
            batchSize,
            clipNorm,
            minLogStd,
            maxLogStd,
            true, // return cost
        );

        // Log metrics
        if (loss1) {
            const critic1LossValue = syncUnwrapTensor(loss1)[0];
            console.info(`[Train Critics]: Critic1 Loss = ${critic1LossValue.toFixed(4)}`);
            metricsChannels.valueLoss.postMessage([critic1LossValue]);

            if (isLossDangerous(critic1LossValue)) {
                console.warn('[Train Critics]: Dangerous critic1 loss detected!');
            }

            loss1.dispose();
        }

        if (loss2) {
            const critic2LossValue = syncUnwrapTensor(loss2)[0];
            console.info(`[Train Critics]: Critic2 Loss = ${critic2LossValue.toFixed(4)}`);

            if (isLossDangerous(critic2LossValue)) {
                console.warn('[Train Critics]: Dangerous critic2 loss detected!');
            }

            loss2.dispose();
        }

        // Soft update target networks
        softUpdateTargetNetwork(critic1, targetCritic1, tau);
        softUpdateTargetNetwork(critic2, targetCritic2, tau);
        console.info('[Train Critics]: Target networks updated');

        // Cleanup tensors
        tStates.forEach(t => t.dispose());
        tNextStates.forEach(t => t.dispose());
        tActions.dispose();
        tRewards.dispose();
        tDones.dispose();

    } finally {
        // Dispose networks
        critic1.dispose();
        critic2.dispose();
        targetCritic1.dispose();
        targetCritic2.dispose();
        actor.dispose();
    }
}
