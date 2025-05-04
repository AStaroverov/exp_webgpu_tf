import * as tf from '@tensorflow/tfjs';
import { Variable } from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { act } from '../../../PPO/train.ts';
import { prepareInputArrays } from '../../InputArrays.ts';
import { disposeNetwork, getNetwork } from '../../../Models/Utils.ts';
import { getNetworkVersion, patientAction } from '../../utils.ts';
import { applyActionToTank } from '../../applyActionToTank.ts';
import { calculateReward } from '../../../Reward/calculateReward.ts';
import { AgentMemory, AgentMemoryBatch } from '../../Memory.ts';
import { getTankHealth } from '../../../../ECS/Entities/Tank/TankUtils.ts';
import { clamp } from 'lodash-es';
import { random, randomRangeFloat } from '../../../../../../../lib/random.ts';
import { Model } from '../../../Models/def.ts';

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

    constructor(public readonly tankEid: number) {
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
            result.actions,
            result.logStd.map((v) => clamp(
                1 - Math.exp(v) / Math.exp(0.2), 0.1, 0.9),
            ),
        );

        const stateReward = calculateReward(
            this.tankEid,
            width,
            height,
        );

        this.memory.addFirstPart(
            state,
            stateReward,
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
        const isDead = getTankHealth(this.tankEid) <= 0;
        const isDone = gameOver || isDead;
        const reward = calculateReward(
            this.tankEid,
            width,
            height,
        );

        this.memory.updateSecondPart(reward, isDone);
    }

    private async load() {
        this.policyNetwork = await getNetwork(Model.Policy);

        if (random() > 0.7) {
            perturbWeights(this.policyNetwork, randomRangeFloat(0.01, 0.03));
        }
    }
}

export function perturbWeights(model: tf.LayersModel, scale: number) {
    tf.tidy(() => {
        model.trainableWeights.forEach(v => {
            const val = v.read() as Variable;
            const std = tf.moments(val).variance.sqrt();
            const eps = tf.randomNormal(val.shape).mul(std).mul(scale);
            val.assign(val.add(eps));
        });
    });
}