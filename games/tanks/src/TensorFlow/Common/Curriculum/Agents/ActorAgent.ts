import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { act } from '../../../PPO/train.ts';
import { prepareInputArrays } from '../../InputArrays.ts';
import { Model } from '../../../Models/Transfer.ts';
import { queueSizeChannel } from '../../../PPO/channels.ts';
import { filter, first, firstValueFrom, mergeMap, race, retry, shareReplay, startWith, tap, timer } from 'rxjs';
import { disposeNetwork, getNetwork } from '../../../Models/Utils.ts';
import { getNetworkVersion } from '../../utils.ts';
import { applyActionToTank } from '../../applyActionToTank.ts';
import { calculateReward } from '../../../Reward/calculateReward.ts';
import { AgentMemory, AgentMemoryBatch } from '../../Memory.ts';
import { getTankHealth } from '../../../../ECS/Entities/Tank/TankUtils.ts';
import { ACTION_DIM } from '../../consts.ts';
import { sqrt } from '../../../../../../../lib/math.ts';

const queueSize$ = queueSizeChannel.obs.pipe(
    startWith(0),
    shareReplay(1),
);
const backpressure$ = race([
    timer(60_000),
    queueSize$.pipe(filter((queueSize) => queueSize < 3)),
]).pipe(first());

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

export class ActorAgent implements TankAgent {
    private noise?: tf.Tensor;
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
        this.noise?.dispose();
        this.noise = undefined;
        this.memory.dispose();
        this.policyNetwork && disposeNetwork(this.policyNetwork);
        this.policyNetwork = undefined;
    }

    public sync() {
        return firstValueFrom(backpressure$.pipe(
            tap(() => this.dispose()),
            mergeMap(() => this.load()),
            retry({ delay: 1000 }),
        ));
    }

    public updateTankBehaviour(
        width: number,
        height: number,
    ) {
        const { noise, std } = this.updateNoise();
        const state = prepareInputArrays(this.tankEid, width, height);
        const result = act(this.policyNetwork!, state, noise, std);

        applyActionToTank(this.tankEid, result.actions);

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
    }

    private updateNoise() {
        const sigma = 0.15;
        const theta = 0.05;
        const std = sigma / sqrt(2 * theta);
        const noise = ouNoise(this.noise, sigma, theta);
        this.noise?.dispose();
        this.noise = noise;

        return { noise, std };
    }
}

function ouNoise(noise: tf.Tensor = tf.zeros([ACTION_DIM]), sigma: number, theta: number) {
    return noise.add(
        tf.randomNormal([ACTION_DIM])
            .mul(sigma)
            .sub(noise.mul(theta)),
    );
}