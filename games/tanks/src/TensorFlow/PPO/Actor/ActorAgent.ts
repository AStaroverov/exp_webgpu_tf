import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { Memory } from '../../Common/Memory.ts';
import { act } from '../train.ts';
import { InputArrays } from '../../Common/InputArrays.ts';
import { Model } from '../../Models/Transfer.ts';
import { queueSizeChannel } from '../channels.ts';
import { filter, first, firstValueFrom, mergeMap, race, retry, shareReplay, startWith, timer } from 'rxjs';
import { getNetworkVersion } from '../../Common/utils.ts';
import { disposeNetwork, getNetwork } from '../../Models/Utils.ts';
import { createPolicyNetwork } from '../../Models/Create.ts';

const queueSize$ = queueSizeChannel.obs.pipe(
    startWith(0),
    shareReplay(1),
);
const backpressure$ = race([
    timer(60_000),
    queueSize$.pipe(filter((queueSize) => queueSize < 2)),
]).pipe(first());

export class ActorAgent {
    private memory: Memory;
    private policyNetwork?: tf.LayersModel;

    constructor() {
        this.memory = new Memory();
    }

    public static create() {
        return new ActorAgent();
    }

    dispose() {
        this.disposeMemory();
    }

    rememberAction(
        tankId: number,
        state: InputArrays,
        stateReward: number,
        action: Float32Array,
        mean: Float32Array,
        logStd: Float32Array,
        logProb: number,
    ) {
        this.memory.addFirstPart(tankId, state, stateReward, action, mean, logStd, logProb);
    }

    rememberReward(tankId: number, reward: number, done: boolean) {
        this.memory.updateSecondPart(tankId, reward, done);
    }

    readMemory() {
        return {
            version: getNetworkVersion(this.policyNetwork!),
            memories: this.memory.getBatch(),
        };
    }

    disposeMemory() {
        this.memory.dispose();
    }

    act(state: InputArrays): {
        actions: Float32Array,
        mean: Float32Array,
        logStd: Float32Array,
        logProb: number,
    } {
        return act(
            this.policyNetwork!,
            state,
        );
    }

    public sync() {
        return firstValueFrom(backpressure$.pipe(
            mergeMap(() => this.load()),
            retry({ delay: 1000 }),
        ));
    }

    private async load() {
        this.resetState();
        this.policyNetwork = await getNetwork(Model.Policy, createPolicyNetwork);
    }

    private resetState() {
        this.policyNetwork && disposeNetwork(this.policyNetwork);
        this.policyNetwork = undefined;
    }
}
