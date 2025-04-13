import '@tensorflow/tfjs-backend-wasm';
import { ceil, floor, max } from '../../../../../../lib/math.ts';
import { CONFIG } from '../config.ts';
import { flatTypedArray } from '../../Common/flat.ts';
import { learnerStateChannel, memoryChannel } from '../../DB';
import { Batch } from '../../Common/Memory.ts';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { PolicyLearnerAgent } from './PolicyLearnerAgent.ts';
import { ValueLearnerAgent } from './ValueLearnerAgent.ts';
import { computeVTraceTargets } from '../train.ts';
import { distinctUntilChanged, filter, first, map, mergeMap, of, shareReplay, withLatestFrom } from 'rxjs';
import { forceExitChannel, metricsChannels } from '../../Common/channels.ts';
import { PrioritizedReplayBuffer } from '../../Common/PrioritizedReplayBuffer.ts';
import { shuffle } from '../../../../../../lib/shuffle.ts';

type ExtendedBatch = (Batch & {
    version: number,
    values: Float32Array,
    returns: Float32Array,
    tdErrors: Float32Array,
    advantages: Float32Array
});

const getIndexes = (length: number) => Array.from({ length }, (_, i) => i);

export class LearnerAgent {
    private policyLA = new PolicyLearnerAgent();
    private valueLA = new ValueLearnerAgent();

    private batches: ExtendedBatch[] = [];
    private lastTrainTimeStart = 0;

    constructor() {
        const isTraining$ = learnerStateChannel.obs.pipe(
            map((state) => state.training),
            distinctUntilChanged(),
            shareReplay(1),
        );

        memoryChannel.obs.pipe(
            withLatestFrom(isTraining$),
            mergeMap(([batch, isTraining]) => {
                return isTraining ? of(batch) : isTraining$.pipe(first((v) => !v), map(() => batch));
            }),
            map((batch): null | ExtendedBatch => {
                const delta = this.getVersion() - batch.version;

                if (delta > 3_000) {
                    console.warn('[Train]: skipping batch with diff', delta);
                    return null;
                } else {
                    return {
                        version: batch.version,
                        ...batch.memories,
                        ...computeVTraceTargets(this.policyLA.network, this.valueLA.network, batch.memories),
                    };
                }
            }),
            filter((batch): batch is ExtendedBatch => batch !== null),
        ).subscribe({
            next: (batch) => {
                this.batches.push(batch);
            },
            error: (error) => {
                console.error('Batch processing:', error);
                forceExitChannel.postMessage(null);
            },
        });
    }

    public train() {
        const startTime = Date.now();
        const waitTime = startTime - (this.lastTrainTimeStart || startTime);
        this.lastTrainTimeStart = startTime;

        const version = this.getVersion();
        const batches = this.batches;
        const batch = getFinalBatch(batches);
        this.batches = [];

        const prb = new PrioritizedReplayBuffer(batch.tdErrors);
        const policyMiniBatchCount = ceil(batch.size / CONFIG.miniBatchSize);
        const valueMiniBatchCount = floor(policyMiniBatchCount / 2);

        console.log(`[Train]: Iteration ${ version },
         Sum batch size: ${ batch.size },
         Mini batch count: ${ policyMiniBatchCount }/${ valueMiniBatchCount } by ${ CONFIG.miniBatchSize }`);

        // policy
        const getPriorPolicyBatch = (batchSize: number) => {
            const { indices, weights } = prb.sample(batchSize);
            return Object.assign(createPolicyBatch(batch, indices), { weights });
        };
        const getRandomKLBatch = (batchSize: number) => {
            return createKlBatch(batch, shuffle(getIndexes(batchSize)));
        };
        const policyTrainMetrics = this.policyLA.train(
            policyMiniBatchCount,
            getPriorPolicyBatch,
            getRandomKLBatch,
            (lr) => {
                this.policyLA.updateLR(lr);
                this.valueLA.updateLR(lr);
                metricsChannels.lr.postMessage(lr);
            },
        );

        // value
        const getValueRandomBatch = (batchSize: number) => {
            return createValueBatch(batch, shuffle(getIndexes(batchSize)));
        };
        const valueTrainMetrics = this.valueLA.train(
            valueMiniBatchCount,
            getValueRandomBatch,
        );

        const endTime = Date.now();

        macroTasks.addTimeout(() => {
            this.logMetrics({
                klList: policyTrainMetrics.klList,
                policyLossList: policyTrainMetrics.policyLossList,
                valueLossList: valueTrainMetrics.valueLossList,
                version,
                policyMiniBatchCount,
                valueMiniBatchCount,
                batches,
                waitTime,
                startTime,
                endTime,
            });
        }, 0);
    }

    public async init() {
        await Promise.all([
            this.policyLA.init(),
            this.valueLA.init(),
        ]);

        return this;
    }

    public async load() {
        return Promise.all([
            this.policyLA.load(),
            this.valueLA.load(),
        ]);
    }

    public async upload() {
        return Promise.all([
            this.policyLA.upload(),
            this.valueLA.upload(),
        ]);
    }

    public async save() {
        while (!(await this.upload())) {
            await new Promise(resolve => macroTasks.addTimeout(resolve, 100));
        }
    }

    public hasEnoughBatches(): boolean {
        return this.batches.reduce((acc, b) => acc + b.size, 0) >= CONFIG.batchSize;
    }

    public healthCheck(): Promise<boolean> {
        return Promise.all([
            this.policyLA.healthCheck(),
            this.valueLA.healthCheck(),
        ]).then(([policy, value]) => policy && value);
    }

    public getVersion() {
        return max(
            this.policyLA.getVersion(),
            this.valueLA.getVersion(),
        );
    }

    private logMetrics(
        {
            klList,
            policyLossList,
            valueLossList,
            version,
            policyMiniBatchCount,
            valueMiniBatchCount,
            batches,
            waitTime,
            startTime,
            endTime,
        }: {
            klList: number[],
            policyLossList: number[],
            valueLossList: number[],
            version: number,
            policyMiniBatchCount: number,
            valueMiniBatchCount: number,
            batches: ExtendedBatch[],
            waitTime: number,
            startTime: number,
            endTime: number,
        },
    ) {

        for (let i = 0; i < klList.length; i++) {
            const kl = klList[i];

            let policyLoss = 0;
            for (let j = 0; j < policyMiniBatchCount; j++) {
                policyLoss += policyLossList[i * policyMiniBatchCount + j];
            }

            policyLoss /= policyMiniBatchCount;

            console.log('[Train]: Epoch', i, 'KL:', kl, 'Policy loss:', policyLoss);

            metricsChannels.kl.postMessage(kl);
            metricsChannels.policyLoss.postMessage(kl);
        }

        for (let i = 0; i < CONFIG.valueEpochs; i++) {
            let valueLoss = 0;
            for (let j = 0; j < valueMiniBatchCount; j++) {
                valueLoss += valueLossList[i * valueMiniBatchCount + j];
            }

            valueLoss /= valueMiniBatchCount;

            console.log('[Train]: Epoch', i, 'Value loss:', valueLoss);

            metricsChannels.valueLoss.postMessage(valueLoss);
        }
        
        for (const batch of batches) {
            metricsChannels.versionDelta.postMessage(version - batch.version);
            metricsChannels.batchSize.postMessage(batch.size);
            metricsChannels.values.postMessage(batch.values);
            metricsChannels.returns.postMessage(batch.returns);
            metricsChannels.advantages.postMessage(batch.advantages);
        }

        metricsChannels.rewards.postMessage(batches.map(b => b.rewards).flat());
        metricsChannels.waitTime.postMessage(waitTime / 1000);
        metricsChannels.trainTime.postMessage((endTime - startTime) / 1000);
    }
}

type FinalBatch = Omit<ExtendedBatch, 'version' | 'dones' | 'rewards'>;

function getFinalBatch(batches: ExtendedBatch[]): FinalBatch {
    return {
        size: batches.reduce((acc, b) => acc + b.size, 0),
        states: batches.map(b => b.states).flat(),
        actions: batches.map(b => b.actions).flat(),
        logProbs: flatTypedArray(batches.map(b => b.logProbs)),
        values: flatTypedArray(batches.map(b => b.values)),
        returns: flatTypedArray(batches.map(b => b.returns)),
        tdErrors: flatTypedArray(batches.map(b => b.tdErrors)),
        advantages: flatTypedArray(batches.map(b => b.advantages)),
    };
}

function createPolicyBatch(batch: FinalBatch, indices: number[]) {
    const states = indices.map(i => batch.states[i]);
    const actions = indices.map(i => batch.actions[i]);
    const logProbs = indices.map(i => batch.logProbs[i]);
    const advantages = indices.map(i => batch.advantages[i]);

    return {
        states: states,
        actions: actions,
        logProbs: (logProbs),
        advantages: (advantages),
    };
}

function createKlBatch(batch: FinalBatch, indices: number[]) {
    const states = indices.map(i => batch.states[i]);
    const actions = indices.map(i => batch.actions[i]);
    const logProbs = indices.map(i => batch.logProbs[i]);

    return {
        states: states,
        actions: actions,
        logProbs: (logProbs),
    };
}

function createValueBatch(batch: FinalBatch, indices: number[]) {
    const states = indices.map(i => batch.states[i]);
    const values = indices.map(i => batch.values[i]);
    const returns = indices.map(i => batch.returns[i]);

    return {
        states: states,
        values: (values),
        returns: (returns),
    };
}

