import '@tensorflow/tfjs-backend-wasm';
import { ceil, max } from '../../../../../../lib/math.ts';
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

type ExtendedBatch = (Batch & {
    version: number,
    values: Float32Array,
    returns: Float32Array,
    tdErrors: Float32Array,
    advantages: Float32Array
});

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
        const miniBatchCount = ceil(batch.size / CONFIG.miniBatchSize);

        console.log(`[Train]: Iteration ${ version },
         Sum batch size: ${ batch.size },
         Mini batch count: ${ miniBatchCount } by ${ CONFIG.miniBatchSize }`);

        // policy
        const getPolicyBatch = (batchSize: number, index: number) => {
            const indices = prb.getSampleWithTop(batchSize, index * batchSize, (index + 1) * batchSize);
            return createPolicyBatch(batch, indices);
        };
        const getKLBatch = (size: number) => {
            return createKlBatch(batch, prb.getSample(batch.size).slice(0, size));
        };
        const policyTrainMetrics = this.policyLA.train(
            miniBatchCount,
            getPolicyBatch,
            getKLBatch,
            (lr) => {
                this.policyLA.updateLR(lr);
                this.valueLA.updateLR(lr);
                metricsChannels.lr.postMessage(lr);
            },
        );

        // value
        const getValueBatch = (batchSize: number, index: number) => {
            return createValueBatch(batch, prb.getSample(batchSize, index * batchSize, (index + 1) * batchSize));
        };
        const valueTrainMetrics = this.valueLA.train(
            miniBatchCount,
            getValueBatch,
        );

        const endTime = Date.now();

        macroTasks.addTimeout(() => {
            this.logMetrics({
                klList: policyTrainMetrics.klList,
                policyLossList: policyTrainMetrics.policyLossList,
                valueLossList: valueTrainMetrics.valueLossList,
                version,
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
            batches,
            waitTime,
            startTime,
            endTime,
        }: {
            klList: number[],
            policyLossList: number[],
            valueLossList: number[],
            version: number,
            batches: ExtendedBatch[],
            waitTime: number,
            startTime: number,
            endTime: number,
        },
    ) {
        console.log('[Metrics] KL', klList.map(v => v.toFixed(4)));
        metricsChannels.kl.postMessage(klList);
        console.log('[Metrics] Policy loss', policyLossList.map(v => v.toFixed(4)));
        metricsChannels.policyLoss.postMessage(policyLossList);
        console.log('[Metrics] Value loss', valueLossList.map(v => v.toFixed(4)));
        metricsChannels.valueLoss.postMessage(valueLossList);

        for (const batch of batches) {
            metricsChannels.versionDelta.postMessage(version - batch.version);
            metricsChannels.batchSize.postMessage(batch.size);
            metricsChannels.rewards.postMessage(batch.rewards);
            metricsChannels.values.postMessage(batch.values);
            metricsChannels.returns.postMessage(batch.returns);
            metricsChannels.tdErrors.postMessage(batch.tdErrors);
            metricsChannels.advantages.postMessage(batch.advantages);
        }

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
        mean: batches.map(b => b.mean).flat(),
        logStd: batches.map(b => b.logStd).flat(),
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
    const mean = indices.map(i => batch.mean[i]);
    const logStd = indices.map(i => batch.logStd[i]);

    return {
        states: states,
        actions: actions,
        mean: (mean),
        logStd: (logStd),
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

