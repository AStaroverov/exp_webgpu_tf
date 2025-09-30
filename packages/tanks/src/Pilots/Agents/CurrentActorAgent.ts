import * as tf from '@tensorflow/tfjs';
import { Variable } from '@tensorflow/tfjs';
import { clamp } from 'lodash-es';
import { RingBuffer } from 'ring-buffer-ts';
import { lerp } from '../../../../../lib/math.ts';
import { random } from '../../../../../lib/random.ts';
import { getTankHealth } from '../../Game/ECS/Entities/Tank/TankUtils.ts';
import { applyActionToTank } from '../../TensorFlow/Common/applyActionToTank.ts';
import { LEARNING_STEPS } from '../../TensorFlow/Common/consts.ts';
import { prepareInputArrays } from '../../TensorFlow/Common/InputArrays.ts';
import { AgentMemory, AgentMemoryBatch } from '../../TensorFlow/Common/Memory.ts';
import { getNetworkVersion, patientAction } from '../../TensorFlow/Common/utils.ts';
import { Model } from '../../TensorFlow/Models/def.ts';
import { disposeNetwork, getNetwork } from '../../TensorFlow/Models/Utils.ts';
import { CONFIG } from '../../TensorFlow/PPO/config.ts';
import { act, MAX_STD_DEV } from '../../TensorFlow/PPO/train.ts';
import { calculateActionReward, calculateStateReward, GAME_OVER_REWARD_MULTIPLIER, getFramePenalty } from '../../TensorFlow/Reward/calculateReward.ts';


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
    getMemory(): AgentMemory;
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
        return this.policyNetwork != null ? getNetworkVersion(this.policyNetwork) : 0;
    }

    public getMemory(): AgentMemory {
        return this.memory;
    }

    public getMemoryBatch(gameOverReward: number): undefined | AgentMemoryBatch {
        return this.memory.getBatch(gameOverReward);
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
            clamp(version / LEARNING_STEPS / 10, 0, 1)
        );
        const actionReward = this.initialActionReward === undefined
            ? 0
            : calculateActionReward(this.tankEid) - this.initialActionReward;
        const stateRewardMultiplier = clamp(1 - (version / LEARNING_STEPS), 0, 1);
        const actionRewardMultiplier = clamp(version / LEARNING_STEPS, 0, 1) - clamp((version - LEARNING_STEPS * 1.5) / LEARNING_STEPS, 0, 1);

        const frameReward = getFramePenalty(frame);
        const deathReward = isDead ? -GAME_OVER_REWARD_MULTIPLIER : 0;
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
        const version = getNetworkVersion(this.policyNetwork);
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

            const val = v.read() as Variable;
            const std = tf.moments(val).variance.sqrt();
            const eps = tf.randomNormal(val.shape).mul(std).mul(scale);
            val.assign(val.add(eps));
        });
    });
}

function isPerturbable(v: tf.LayerVariable) {
    // 1. работаем только с float-весами
    if (v.dtype !== 'float32') return false;

    // 2. не трогаем BatchNorm-параметры
    if (v.name.includes('batch_normalization') ||
        v.name.includes('batchnorm') ||
        v.name.includes('/gamma') ||  // scale
        v.name.includes('/beta'))     // shift
        return false;

    // 3. (опционально) пропускаем bias-векторы
    if (v.name.endsWith('/bias')) return false;

    return true;
}