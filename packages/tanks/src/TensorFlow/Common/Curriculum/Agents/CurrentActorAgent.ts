import * as tf from '@tensorflow/tfjs';
import { Variable } from '@tensorflow/tfjs';
import { act, MAX_STD_DEV } from '../../../PPO/train.ts';
import { prepareInputArrays } from '../../InputArrays.ts';
import { disposeNetwork, getNetwork } from '../../../Models/Utils.ts';
import { getNetworkVersion, patientAction } from '../../utils.ts';
import { applyActionToTank } from '../../applyActionToTank.ts';
import { calculateActionReward, calculateStateReward } from '../../../Reward/calculateReward.ts';
import { AgentMemory, AgentMemoryBatch } from '../../Memory.ts';
import { getTankHealth } from '../../../../Game/ECS/Entities/Tank/TankUtils.ts';
import { Model } from '../../../Models/def.ts';
import { random } from '../../../../../../../lib/random.ts';
import { CONFIG } from '../../../PPO/config.ts';
import { clamp } from 'lodash-es';
import { lerp } from '../../../../../../../lib/math.ts';

export type TankAgent = {
    tankEid: number;

    sync?(): Promise<void>;
    dispose?(): void;
    getVersion?(): number;

    getMemory?(): AgentMemory;
    getMemoryBatch?(): AgentMemoryBatch;

    updateTankBehaviour(width: number, height: number): void;
    evaluateTankBehaviour?(width: number, height: number, gameOver: boolean): void;
}

export class CurrentActorAgent implements TankAgent {
    private memory = new AgentMemory();
    private policyNetwork?: tf.LayersModel;

    private initialActionReward: undefined | number;

    constructor(public readonly tankEid: number, private train: boolean) {
    }

    public getVersion() {
        return this.policyNetwork != null ? getNetworkVersion(this.policyNetwork) : 0;
    }

    public getMemory() {
        return this.memory;
    }

    public getMemoryBatch() {
        return this.memory.getBatch();
    }

    public dispose() {
        this.initialActionReward = undefined;
        this.policyNetwork && disposeNetwork(this.policyNetwork);
        this.policyNetwork = undefined;
        this.memory.dispose();
    }

    public sync() {
        this.dispose();
        return patientAction(() => this.load());
    }

    public updateTankBehaviour(
        width: number,
        height: number,
    ) {
        const state = prepareInputArrays(this.tankEid, width, height);
        const result = act(this.policyNetwork!, state);

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
        gameOver: boolean,
    ) {
        if (!this.train) return;

        const isDead = getTankHealth(this.tankEid) <= 0;
        const isDone = gameOver || isDead;
        const stateReward = calculateStateReward(
            this.tankEid,
            width,
            height,
        );
        const actionReward = this.initialActionReward === undefined
            ? 0
            : calculateActionReward(this.tankEid) - this.initialActionReward;

        this.memory.updateSecondPart(clamp(stateReward + actionReward, -200, 200), isDone);
    }

    private async load() {
        this.policyNetwork = await getNetwork(Model.Policy);
        if (CONFIG.perturbWeightsScale > 0 && random() > Math.pow(1 - 0.5, 1 / CONFIG.workerCount)) { // 50% for N workers
            perturbWeights(this.policyNetwork, CONFIG.perturbWeightsScale);
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