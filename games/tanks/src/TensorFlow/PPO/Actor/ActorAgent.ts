import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { Memory } from '../../Common/Memory.ts';
import { act } from '../train.ts';
import { InputArrays } from '../../Common/InputArrays.ts';
import { setModelState } from '../../Common/modelsCopy.ts';
import { createPolicyNetwork, createValueNetwork } from '../../Models/Create.ts';
import { loadNetworkFromDB, Model } from '../../Models/Transfer.ts';
import { disposeNetwork } from '../../Models/Utils.ts';
import { learnerStateChannel } from '../../DB';
import {
    distinctUntilChanged,
    filter,
    firstValueFrom,
    forkJoin,
    map,
    mergeMap,
    Observable,
    of,
    retry,
    shareReplay,
    switchMap,
    take,
    tap,
    timer,
} from 'rxjs';

export class ActorAgent {
    private memory: Memory;

    private version = -1;

    private policyNetwork?: tf.LayersModel;
    private valueNetwork?: tf.LayersModel;

    private learnerState$: Observable<{ version: number, training: boolean }>;
    private backpressure$: Observable<boolean>;
    private hasNewNetworks$: Observable<boolean>;

    constructor() {
        this.memory = new Memory();

        this.learnerState$ = learnerStateChannel.obs.pipe(
            shareReplay(1),
        );
        this.backpressure$ = this.learnerState$.pipe(
            map((state) => state.training),
            switchMap((shouldWait) => shouldWait ? timer(8_000).pipe(map(() => true)) : of(false)),
            distinctUntilChanged(),
            shareReplay(1),
        );
        this.hasNewNetworks$ = this.learnerState$.pipe(
            map((states) => states.version > this.version),
            shareReplay(1),
        );
    }

    public static create() {
        return new ActorAgent();
    }

    dispose() {
        this.disposeMemory();
    }

    rememberAction(tankId: number, state: InputArrays, action: Float32Array, logProb: number) {
        this.memory.addFirstPart(tankId, state, action, logProb);
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
        logProb: number,
    } {
        if (this.policyNetwork == null || this.valueNetwork == null) {
            throw new Error('Models not loaded');
        }

        return act(
            this.policyNetwork,
            state,
        );
    }

    public sync() {
        return firstValueFrom(this.backpressure$.pipe(
            filter((shouldWait) => !shouldWait),
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
        return this.policyNetwork == null || this.valueNetwork == null;
    }

    private load() {
        return forkJoin([loadNetworkFromDB(Model.Value), loadNetworkFromDB(Model.Policy)]).pipe(
            mergeMap(([valueNetwork, policyNetwork]) => {
                if (!valueNetwork || !policyNetwork) {
                    throw new Error('Models not loaded');
                }

                return forkJoin([
                    this.learnerState$.pipe(take(1)),
                    setModelState(this.policyNetwork ?? createPolicyNetwork(), policyNetwork),
                    setModelState(this.valueNetwork ?? createValueNetwork(), valueNetwork),
                ]).pipe(
                    tap({
                        error: () => {

                            this.resetNetworks();
                        },
                        finalize: () => {
                            disposeNetwork(policyNetwork);
                            disposeNetwork(valueNetwork);
                        },
                    }),
                );
            }),
            tap(([state, policyNetwork, valueNetwork]) => {
                this.version = state.version;
                this.policyNetwork = policyNetwork;
                this.valueNetwork = valueNetwork;
            }),
        );
    }

    private resetNetworks() {
        this.policyNetwork?.dispose();
        this.valueNetwork?.dispose();
        this.policyNetwork = undefined;
        this.valueNetwork = undefined;
        this.version = -1;
    }
}
