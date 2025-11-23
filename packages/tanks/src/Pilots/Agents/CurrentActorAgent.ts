import * as tf from '@tensorflow/tfjs';
import { clamp } from 'lodash-es';
import { unlerp } from '../../../../../lib/math.ts';
import { randomRangeFloat } from '../../../../../lib/random.ts';
import { applyActionToTank } from '../../../../ml-common/applyActionToTank.ts';
import { ColoredNoiseApprox } from '../../../../ml-common/ColoredNoiseApprox.ts';
import { LEARNING_STEPS } from '../../../../ml-common/consts.ts';
import { prepareInputArrays } from '../../../../ml-common/InputArrays.ts';
import { AgentMemory, AgentMemoryBatch } from '../../../../ml-common/Memory.ts';
import { getNetworkExpIteration, patientAction } from '../../../../ml-common/utils.ts';
import { ACTION_HEAD_DIMS } from '../../../../ml/src/Models/Create.ts';
import { Model } from '../../../../ml/src/Models/def.ts';
import { disposeNetwork, getNetwork } from '../../../../ml/src/Models/Utils.ts';
import { noisyAct } from '../../../../ml/src/PPO/train.ts';
import { calculateActionReward, calculateStateReward, getDeathPenalty, getFramePenalty } from '../../../../ml/src/Reward/calculateReward.ts';
import { getTankHealth } from '../../Game/ECS/Entities/Tank/TankUtils.ts';

export type TankAgent<A = Partial<DownloableAgent> & Partial<LearnableAgent>> = A & {
    tankEid: number;
    updateTankBehaviour(width: number, height: number, frame: number): void;
}

export type DownloableAgent = {
    dispose(): void;
    sync(): Promise<void>;
    isSynced(): boolean;
}

export type LearnableAgent = {
    dispose(): void;
    getVersion(): number;
    getMemory(): undefined | AgentMemory;
    getMemoryBatch(gameOverReward: number): undefined | AgentMemoryBatch;
    evaluateTankBehaviour(width: number, height: number, frame: number): void;
}

export class CurrentActorAgent implements TankAgent<DownloableAgent & LearnableAgent> {
    private memory = new AgentMemory();
    private noise?: ColoredNoiseApprox;
    private policyNetwork?: tf.LayersModel;

    private initialActionReward?: number;

    constructor(public readonly tankEid: number, private train: boolean) {
    }

    public getVersion(): number {
        return this.policyNetwork != null ? getNetworkExpIteration(this.policyNetwork) : 0;
    }

    public getMemory(): undefined | AgentMemory {
        return this.train ? this.memory : undefined;
    }

    public getMemoryBatch(gameOverReward: number): undefined | AgentMemoryBatch {
        return this.train ? this.memory.getBatch(gameOverReward) : undefined;
    }

    public dispose() {
        disposeNetwork(this.policyNetwork);
        this.memory.dispose();
        this.noise?.dispose();
    }

    public async sync() {
        await patientAction(() => this.load());
    }

    public isSynced() {
        return this.policyNetwork != null;
    }

    public updateTankBehaviour(
        width: number,
        height: number,
    ) {
        if (this.policyNetwork == null) return;

        const state = prepareInputArrays(this.tankEid, width, height);
        const useNoise =
            this.noise != null
            // @ts-ignore
            && globalThis.disableNoise !== true;
        const result = noisyAct(
            this.policyNetwork,
            state,
            useNoise ? this.noise?.sample() : undefined
        );

        applyActionToTank(this.tankEid, result.actions, false);

        if (!this.train) return;

        this.initialActionReward = calculateActionReward(this.tankEid);

        this.memory.addFirstPart(
            state,
            result.actions,
            result.logits,
            result.logProb,
        );
    }

    public evaluateTankBehaviour(
        width: number,
        height: number,
        frame: number,
    ) {
        if (!this.train || this.memory.size() === 0) return;

        const isDead = getTankHealth(this.tankEid) <= 0;
        const version = this.getVersion();
        const stateReward = calculateStateReward(
            this.tankEid,
            width,
            height,
            clamp(version / (LEARNING_STEPS * 0.2), 0, 1)
        );
        const actionReward = this.initialActionReward === undefined
            ? 0
            : calculateActionReward(this.tankEid) - this.initialActionReward;
        const stateRewardMultiplier = clamp(1 - unlerp(0, LEARNING_STEPS * 0.4, version), 0, 0.5);
        const actionRewardMultiplier = clamp(unlerp(0, LEARNING_STEPS * 0.2, version), 0.3, 1);// - clamp(unlerp(LEARNING_STEPS * 0.6, LEARNING_STEPS, version), 0, 0.5);

        const frameReward = getFramePenalty(frame);
        const deathReward = getDeathPenalty(isDead);
        // it's not all reward, also we have final reward for lose/win in the end of episode
        const reward = clamp(
            stateReward * stateRewardMultiplier
            + actionReward * actionRewardMultiplier
            + frameReward
            + deathReward,
            -5,
            +5
        );

        this.memory.updateSecondPart(reward, isDead);
    }

    private async load() {
        this.policyNetwork = await getNetwork(Model.Policy);

        if (this.train || globalThis.document !== undefined) {
            this.noise = new ColoredNoiseApprox(ACTION_HEAD_DIMS, randomRangeFloat(0.3, 0.7));
        }
    }
}

export function perturbWeights(model: tf.LayersModel, scale: number) {
    tf.tidy(() => {
        for (const v of model.trainableWeights) {
            if (!isPerturbable(v)) continue;

            const val = v.read() as tf.Tensor;

            let eps: tf.Tensor;
            if (val.shape.length === 2) {
                const [outDim, inDim] = val.shape;
                const epsOut = tf.randomNormal([outDim, 1]);
                const epsIn = tf.randomNormal([1, inDim]);
                eps = epsOut.mul(epsIn);
            } else {
                eps = tf.randomNormal(val.shape);
            }

            const std = tf.moments(val).variance.sqrt();
            const perturbed = val.add(eps.mul(std).mul(scale));
            (val as tf.Variable).assign(perturbed);
        }
    });
}

function isPerturbable(v: tf.LayerVariable) {
    // 1. Только float32 параметры
    if (v.dtype !== 'float32') return false;

    // 2. Исключаем BatchNorm (они чувствительны к шуму)
    const name = v.name.toLowerCase();
    if (
        name.includes('batchnorm') ||
        name.includes('batch_normalization') ||
        name.includes('layernorm') ||
        name.includes('layer_norm') ||
        name.endsWith('/gamma') ||
        name.endsWith('/beta') ||
        name.endsWith('/moving_mean') ||
        name.endsWith('/moving_variance')
    ) {
        return false;
    }

    return name.includes('mean_mlp');
}
