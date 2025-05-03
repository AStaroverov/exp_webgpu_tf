import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { act, ouNoise, perturbWeights } from '../../../PPO/train.ts';
import { prepareInputArrays } from '../../InputArrays.ts';
import { Model } from '../../../Models/Transfer.ts';
import { queueSizeChannel } from '../../../PPO/channels.ts';
import { filter, first, firstValueFrom, mergeMap, race, retry, shareReplay, startWith, tap, timer } from 'rxjs';
import { disposeNetwork, getNetwork } from '../../../Models/Utils.ts';
import { getNetworkVersion } from '../../utils.ts';
import { Actions, applyActionToTank } from '../../applyActionToTank.ts';
import { calculateReward } from '../../../Reward/calculateReward.ts';
import { AgentMemory, AgentMemoryBatch } from '../../Memory.ts';
import { getTankHealth } from '../../../../ECS/Entities/Tank/TankUtils.ts';
import { ACTION_DIM } from '../../consts.ts';
import { max, mean } from '../../../../../../../lib/math.ts';

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
    private step = 0;
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
        this.policyNetwork && disposeNetwork(this.policyNetwork);
        this.policyNetwork = undefined;
        this.noise?.dispose();
        this.noise = undefined;
        this.memory.dispose();
        this.step = 0;
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
        const state = prepareInputArrays(this.tankEid, width, height);
        const result = act(this.policyNetwork!, state, this.noise);

        if (this.step++ % 30 === 0) {
            const sigma = max(2 * mean(result.logStd.map(Math.exp)), 0.05);
            const newNoise = ouNoise(this.noise ?? tf.zeros([ACTION_DIM]), sigma);
            this.noise?.dispose();
            this.noise = newNoise;
        }

        applyActionToTank(this.tankEid, result.actions.map(v => v / 10) as Actions);

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
        perturbWeights(this.policyNetwork);
    }
}
