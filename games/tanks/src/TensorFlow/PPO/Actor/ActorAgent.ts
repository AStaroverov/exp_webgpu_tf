import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { Memory } from '../../Common/Memory.ts';
import { CONFIG } from '../config.ts';
import { act } from '../train.ts';
import { InputArrays } from '../../Common/InputArrays.ts';
import { setModelState } from '../../Common/modelsCopy.ts';
import { createPolicyNetwork, createValueNetwork } from '../../Models/Create.ts';
import { loadNetworkFromDB, Model } from '../../Models/Transfer.ts';
import { disposeNetwork } from '../../Models/Utils.ts';
import { learnerStateChannel } from '../../DB';
import {
    filter,
    firstValueFrom,
    forkJoin,
    map,
    mergeMap,
    Observable,
    of,
    retry,
    scan,
    shareReplay,
    take,
    tap,
} from 'rxjs';

export class ActorAgent {
    private memory: Memory;

    private policyVersion = -1;
    private valueVersion = -1;

    private policyNetwork?: tf.LayersModel;
    private valueNetwork?: tf.LayersModel;

    private agentsState$: Observable<Record<Model, { version: number, training: boolean }>>;
    private hasNewNetworks$: Observable<boolean>;
    private shouldWaitLearnerTraining$: Observable<boolean>;

    constructor() {
        this.memory = new Memory();

        this.agentsState$ = learnerStateChannel.obs.pipe(
            scan((acc, { model, version, training }) => {

                acc[model] = { version, training };
                return acc;
            }, {} as Record<Model, { version: number, training: boolean }>),
            filter((states) => {

                return states[Model.Policy] != null && states[Model.Value] != null;
            }),
            shareReplay(1),
        );
        this.shouldWaitLearnerTraining$ = this.agentsState$.pipe(
            map((states) => {

                return states[Model.Policy].training || states[Model.Value].training;
            }),
        );
        this.hasNewNetworks$ = this.agentsState$.pipe(
            map((states) => states[Model.Policy].version > this.policyVersion && states[Model.Value].version > this.valueVersion),
        );
    }

    public static create() {
        return new ActorAgent();
    }

    dispose() {
        this.disposeMemory();
    }

    rememberAction(tankId: number, state: InputArrays, action: Float32Array, logProb: number, value: number) {
        this.memory.addFirstPart(tankId, state, action, logProb, value);
    }

    rememberReward(tankId: number, reward: number, done: boolean, isLast = false) {
        this.memory.updateSecondPart(tankId, reward, done, isLast);
    }

    readMemory() {
        return {
            version: {
                [Model.Policy]: this.policyVersion,
                [Model.Value]: this.valueVersion,
            },
            memories: this.memory.getBatch(CONFIG.gamma, CONFIG.lam),
        };
    }

    disposeMemory() {
        this.memory.dispose();
    }

    act(state: InputArrays): {
        actions: Float32Array,
        logProb: number,
        value: number
    } {
        if (this.policyNetwork == null || this.valueNetwork == null) {
            throw new Error('Models not loaded');
        }

        return act(
            this.policyNetwork,
            this.valueNetwork,
            state,
        );
    }

    public sync() {
        return firstValueFrom(this.shouldWaitLearnerTraining$.pipe(
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
                    this.agentsState$.pipe(take(1)),
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
            tap(([agentsStates, policyNetwork, valueNetwork]) => {
                this.policyVersion = agentsStates[Model.Policy].version;
                this.policyNetwork = policyNetwork;

                this.valueVersion = agentsStates[Model.Value].version;
                this.valueNetwork = valueNetwork;
            }),
        );
    }

    private resetNetworks() {
        this.policyNetwork?.dispose();
        this.valueNetwork?.dispose();
        this.policyNetwork = undefined;
        this.valueNetwork = undefined;
        this.policyVersion = -1;
        this.valueVersion = -1;
    }
}
