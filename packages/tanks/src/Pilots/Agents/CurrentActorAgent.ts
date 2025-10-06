import * as tf from '@tensorflow/tfjs';
import { clamp } from 'lodash-es';
import { RingBuffer } from 'ring-buffer-ts';
import { abs, unlerp } from '../../../../../lib/math.ts';
import { random } from '../../../../../lib/random.ts';
import { applyActionToTank } from '../../../../ml-common/applyActionToTank.ts';
import { CONFIG } from '../../../../ml-common/config.ts';
import { LEARNING_STEPS } from '../../../../ml-common/consts.ts';
import { prepareInputArrays } from '../../../../ml-common/InputArrays.ts';
import { AgentMemory, AgentMemoryBatch } from '../../../../ml-common/Memory.ts';
import { getNetworkExpIteration, getNetworkPerturbScale, patientAction } from '../../../../ml-common/utils.ts';
import { Model } from '../../../../ml/src/Models/def.ts';
import { disposeNetwork, getNetwork } from '../../../../ml/src/Models/Utils.ts';
import { act } from '../../../../ml/src/PPO/train.ts';
import { calculateActionReward, calculateStateReward, getDeathPenalty, getFramePenalty } from '../../../../ml/src/Reward/calculateReward.ts';
import { getTankHealth } from '../../Game/ECS/Entities/Tank/TankUtils.ts';


let stateRewardHistory = new RingBuffer<number>(1000);
let actionRewardHistory = new RingBuffer<number>(1000);

setInterval(() => {
    const stateRewards = stateRewardHistory.toArray();
    const actionRewards = actionRewardHistory.toArray();

    let stateMin = Infinity;
    let stateMax = -Infinity;
    let stateAvg = 0;
    for (const v of stateRewards) {
        if (v < stateMin) stateMin = v;
        if (v > stateMax) stateMax = v;
        stateAvg += abs(v);
    }
    stateAvg /= stateRewards.length;

    let actionMin = Infinity;
    let actionMax = -Infinity;
    let actionAvg = 0;
    for (const v of actionRewards) {
        if (v < actionMin) actionMin = v;
        if (v > actionMax) actionMax = v;
        actionAvg += abs(v);
    }
    actionAvg /= actionRewards.length;

    console.log('Avg rewards:', `
        state min=${stateMin.toFixed(2)} max=${stateMax.toFixed(2)} avg=${stateAvg.toFixed(2)}
        action min=${actionMin.toFixed(2)} max=${actionMax.toFixed(2)} avg=${actionAvg.toFixed(2)}
    `);
}, 60 * 1000)

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
    private policyNetwork?: tf.LayersModel;

    private initialActionReward: undefined | number;

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
        this.initialActionReward = undefined;
        this.policyNetwork && disposeNetwork(this.policyNetwork);
        this.policyNetwork = undefined;
        this.memory.dispose();
    }

    public async sync() {
        this.dispose();
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
        const result = act(this.policyNetwork, state);

        applyActionToTank(
            this.tankEid,
            result.actions.map((v) => clamp(v, -1, 1)),
            // result.logStd.map((v) => lerp(0.1, 1, 1 - Math.exp(v) / MAX_STD_DEV)),
        );

        if (!this.train) return;

        this.initialActionReward = calculateActionReward(this.tankEid);

        this.memory.addFirstPart(
            state,
            result.actions,
            result.mean,
            result.logStd,
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
        const stateRewardMultiplier = clamp(1 - unlerp(0, LEARNING_STEPS * 0.4, version), 0, 1);
        const actionRewardMultiplier = clamp(unlerp(0, LEARNING_STEPS * 0.2, version), 0.2, 1) - clamp(unlerp(LEARNING_STEPS * 0.6, LEARNING_STEPS, version), 0, 1);

        const frameReward = getFramePenalty(frame);
        const deathReward = getDeathPenalty(isDead);
        // it's not all reward, also we have final reward for lose/win in the end of episode
        const reward = clamp(
            stateReward * stateRewardMultiplier
            + actionReward * actionRewardMultiplier
            + frameReward
            + deathReward,
            -30,
            +30
        );

        this.memory.updateSecondPart(reward, isDead);

        stateRewardHistory.add(stateReward);
        actionRewardHistory.add(actionReward);
    }

    private async load() {
        this.policyNetwork = await getNetwork(Model.Policy);
        const version = getNetworkExpIteration(this.policyNetwork);
        const chance = CONFIG.perturbChance(version);

        if (chance > random()) {
            this.memory.perturbed = true;
            perturbWeights(this.policyNetwork, getNetworkPerturbScale(this.policyNetwork));
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
                // Матрица весов [out_dim, in_dim]
                const [outDim, inDim] = val.shape;
                const epsOut = tf.randomNormal([outDim, 1]);
                const epsIn = tf.randomNormal([1, inDim]);
                // Двухфакторный шум (снижает рандом до более "структурированного" уровня)
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

    return true;
}