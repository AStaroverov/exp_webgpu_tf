import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { Memory } from '../../Common/Memory.ts';
import { act } from '../train.ts';
import { InputArrays } from '../../Common/InputArrays.ts';
import { setModelState } from '../../Common/modelsCopy.ts';
import { createPolicyNetwork } from '../../Models/Create.ts';
import { loadNetworkFromDB, Model } from '../../Models/Transfer.ts';
import { learnerStateChannel } from '../channels.ts';
import { filter, firstValueFrom, map, mergeMap, Observable, of, retry, shareReplay, startWith, take, tap } from 'rxjs';
import { fromPromise } from 'rxjs/internal/observable/innerFrom';
import { getNetworkVersion } from '../../Common/utils.ts';

export class ActorAgent {
    private memory: Memory;

    private version = -1;
    private policyNetwork?: tf.LayersModel;

    private backpressure$: Observable<unknown>;
    private hasNewNetworks$: Observable<boolean>;

    constructor() {
        this.memory = new Memory();

        const learnerState$ = learnerStateChannel.obs.pipe(
            startWith({ version: 0, queueSize: 0, training: false }),
            shareReplay(1),
        );

        this.hasNewNetworks$ = learnerState$.pipe(
            map((states) => states.version > this.version),
        );
        this.backpressure$ = learnerState$.pipe(
            filter((states) => states.queueSize < 3),
        );

        // hot observable
        learnerState$.subscribe();
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
            version: this.version,
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
        if (this.policyNetwork == null) {
            throw new Error('Models not loaded');
        }

        return act(
            this.policyNetwork,
            state,
        );
    }

    public sync() {
        return firstValueFrom(this.backpressure$.pipe(
            mergeMap(() => this.hasNewNetworks$),
            mergeMap((hasNew) => {
                if (!hasNew) return of(this.shouldInitNetworks());
                return this.load().pipe(map(() => false));
            }),
            filter((shouldWait) => !shouldWait),
            retry({ delay: 1000 }),
            take(1),
        ));
    }

    private shouldInitNetworks() {
        return this.policyNetwork == null;
    }

    private load() {
        return fromPromise(loadNetworkFromDB(Model.Policy)).pipe(
            mergeMap((policyNetwork) => {
                if (!policyNetwork) {
                    throw new Error('Models not loaded');
                }

                return setModelState(this.policyNetwork ?? createPolicyNetwork(), policyNetwork).finally(() => {
                    policyNetwork.dispose();
                });
            }),
            tap({
                next: (policyNetwork) => {
                    this.version = getNetworkVersion(policyNetwork);
                    this.policyNetwork = policyNetwork;
                    console.log('Models loaded successfully');
                },
                error: () => this.resetState(),
            }),
        );
    }

    private resetState() {
        this.policyNetwork?.dispose();
        this.policyNetwork = undefined;
        this.version = -1;
    }
}
