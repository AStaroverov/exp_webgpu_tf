import * as tf from '@tensorflow/tfjs';
import { clamp } from 'lodash-es';
import { RingBuffer } from 'ring-buffer-ts';
import { lerp, unlerp } from '../../../../../lib/math.ts';
import { random } from '../../../../../lib/random.ts';
import { applyActionToTank } from '../../../../ml-common/applyActionToTank.ts';
import { LEARNING_STEPS } from '../../../../ml-common/consts.ts';
import { prepareInputArrays } from '../../../../ml-common/InputArrays.ts';
import { AgentMemory, AgentMemoryBatch } from '../../../../ml-common/Memory.ts';
import { getNetworkExpIteration, patientAction } from '../../../../ml-common/utils.ts';
import { Model } from '../../../../ml/src/Models/def.ts';
import { disposeNetwork, getNetwork } from '../../../../ml/src/Models/Utils.ts';
import { CONFIG } from '../../../../ml/src/PPO/config.ts';
import { act, MAX_STD_DEV } from '../../../../ml/src/PPO/train.ts';
import { calculateActionReward, calculateStateReward, getDeathPenalty, getFramePenalty } from '../../../../ml/src/Reward/calculateReward.ts';
import { getTankHealth } from '../../Game/ECS/Entities/Tank/TankUtils.ts';


let stateRewardHistory = new RingBuffer<number>(1000);
let actionRewardHistory = new RingBuffer<number>(1000);

setInterval(() => {
    const stateRewards = stateRewardHistory.toArray();
    const actionRewards = actionRewardHistory.toArray();

    let stateMin = Infinity;
    let stateMax = -Infinity;
    for (const v of stateRewards) {
        if (v < stateMin) stateMin = v;
        if (v > stateMax) stateMax = v;
    }

    let actionMin = Infinity;
    let actionMax = -Infinity;
    for (const v of actionRewards) {
        if (v < actionMin) actionMin = v;
        if (v > actionMax) actionMax = v;
    }

    console.log('Avg rewards:', `
        state min=${stateMin.toFixed(2)} max=${stateMax.toFixed(2)}
        action min=${actionMin.toFixed(2)} max=${actionMax.toFixed(2)}
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
            result.logStd.map((v) => lerp(0.1, 1, 1 - Math.exp(v) / MAX_STD_DEV)),
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
            clamp(version / (LEARNING_STEPS / 10), 0, 1)
        );
        const actionReward = this.initialActionReward === undefined
            ? 0
            : calculateActionReward(this.tankEid) - this.initialActionReward;
        const stateRewardMultiplier = clamp(1 - unlerp(0, LEARNING_STEPS * 0.2, version), 0, 1);
        const actionRewardMultiplier = clamp(unlerp(0, LEARNING_STEPS * 0.2, version), 0.2, 1) - clamp(unlerp(LEARNING_STEPS * 0.6, LEARNING_STEPS, version), 0, 1);


        const frameReward = getFramePenalty(frame);
        const deathReward = getDeathPenalty(isDead);
        // it's not all reward, also we have final reward for lose/win in the end of episode
        const reward = clamp(
            stateReward * stateRewardMultiplier
            + actionReward * actionRewardMultiplier
            + frameReward
            + deathReward,
            -100,
            100
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
            perturbWeights(this.policyNetwork, CONFIG.perturbWeightsScale(version));
        }
    }
}

export function perturbWeights(model: tf.LayersModel, scale: number) {
    tf.tidy(() => {
        model.trainableWeights.forEach(v => {
            if (!isPerturbable(v)) return;

            const val = v.read() as tf.Tensor; // веса (или bias)
            let eps: tf.Tensor;

            if (val.shape.length === 2) {
                // === Матрица весов [out_dim, in_dim] ===
                const [outDim, inDim] = val.shape;
                const epsOut = tf.randomNormal([outDim, 1]); // шум для строк
                const epsIn = tf.randomNormal([1, inDim]);   // шум для столбцов
                eps = epsOut.mul(epsIn).mul(scale);
            } else {
                eps = tf.randomNormal(val.shape).mul(scale);
            }

            const perturbed = val.add(eps);
            (val as tf.Variable).assign(perturbed);
        });
    });
}

function isPerturbable(v: tf.LayerVariable) {
    // 1. только float32-параметры
    if (v.dtype !== 'float32') return false;

    // 2. исключаем BatchNorm параметры
    if (v.name.includes('batch_normalization') ||
        v.name.includes('batchnorm') ||
        v.name.includes('/gamma') ||  // scale
        v.name.includes('/beta'))     // shift
        return false;

    return true;
}
