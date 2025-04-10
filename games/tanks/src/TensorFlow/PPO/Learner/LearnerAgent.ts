import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { ACTION_DIM } from '../../Common/consts.ts';
import { ceil, max } from '../../../../../../lib/math.ts';
import { CONFIG } from '../config.ts';
import { flatFloat32Array } from '../../Common/flat.ts';
import { batchShuffle } from '../../../../../../lib/shuffle.ts';
import { createInputTensors } from '../../Common/InputTensors.ts';
import { learnerStateChannel, memoryChannel } from '../../DB';
import { Batch } from '../../Common/Memory.ts';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { PolicyLearnerAgent } from './PolicyLearnerAgent.ts';
import { ValueLearnerAgent } from './ValueLearnerAgent.ts';
import { computeVTraceTargets } from '../train.ts';
import { distinctUntilChanged, first, map, mergeMap, of, shareReplay, withLatestFrom } from 'rxjs';
import { metricsChannels } from '../../Common/channels.ts';

type ExtendedBatch = (Batch & {
    version: number,
    values: Float32Array,
    returns: Float32Array,
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
        ).subscribe((batch) => {
            const delta = this.getVersion() - batch.version;

            if (delta > 3_000) {
                console.warn('[Train]: skipping batch with diff', delta);
            } else {
                this.batches.push({
                    version: batch.version,
                    ...batch.memories,
                    ...computeVTraceTargets(this.policyLA.network, this.valueLA.network, batch.memories),
                });
            }
        });
    }

    public train() {
        const startTime = Date.now();
        const waitTime = startTime - (this.lastTrainTimeStart || startTime);
        this.lastTrainTimeStart = startTime;

        const version = this.getVersion();
        const batches = this.batches;
        this.batches = [];

        const batchSize = batches.reduce((acc, b) => acc + b.size, 0);
        const states = batches.map(b => b.states).flat();
        const actions = batches.map(b => b.actions).flat();
        const logProbs = new Float32Array(batches.map(b => b.logProbs).flat());
        const values = flatFloat32Array(batches.map(b => b.values));
        const returns = flatFloat32Array(batches.map(b => b.returns));
        const advantages = flatFloat32Array(batches.map(b => b.advantages));
        debugger

        const miniBatchCount = ceil(states.length / CONFIG.miniBatchSize);
        const miniBatchIndexes = Array.from({ length: miniBatchCount }, (_, i) => i);

        batchShuffle(
            states,
            actions,
            logProbs,
            values,
            returns,
            advantages,
        );

        const tAllStates = createInputTensors(states);
        const tAllActions = tf.tensor2d(flatFloat32Array(actions), [actions.length, ACTION_DIM]);
        const tAllLogProbs = tf.tensor1d(logProbs);
        const tAllValues = tf.tensor1d(values);
        const tAllReturns = tf.tensor1d(returns);
        const tAllAdvantages = tf.tensor1d(advantages);

        console.log(`[Train]: Iteration ${ version }, Sum batch size: ${ batchSize }, Mini batch count: ${ miniBatchCount } by ${ CONFIG.miniBatchSize }`);

        const policyTrainMetrics = this.policyLA.train(
            batchSize,
            miniBatchIndexes,
            tAllStates,
            tAllActions,
            tAllLogProbs,
            tAllAdvantages,
            (lr) => {
                this.policyLA.updateLR(lr);
                this.valueLA.updateLR(lr);
                metricsChannels.lr.postMessage(lr);
            },
        );
        const valueTrainMetrics = this.valueLA.train(
            batchSize,
            miniBatchIndexes,
            tAllStates,
            tAllValues,
            tAllReturns,
        );

        this.logMetrics({
            policyTrainMetrics,
            valueTrainMetrics,
            version,
            miniBatchCount,
            batches,
            values,
            returns,
            advantages,
            waitTime,
            startTime,
            endTime: Date.now(),
        });

        tAllStates.forEach(t => t.dispose());
        tAllActions.dispose();
        tAllLogProbs.dispose();
        tAllAdvantages.dispose();
        tAllValues.dispose();
        tAllReturns.dispose();
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
        return this.batches.length >= CONFIG.workerCount / 2;
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
            policyTrainMetrics,
            valueTrainMetrics,
            version,
            miniBatchCount,
            batches,
            values,
            returns,
            advantages,
            waitTime,
            startTime,
            endTime,
        }: {
            policyTrainMetrics: Promise<{ klList: number[], policyLossList: number[] }>,
            valueTrainMetrics: Promise<{ valueLossList: number[] }>,
            version: number,
            miniBatchCount: number,
            batches: ExtendedBatch[],
            values: Float32Array,
            returns: Float32Array,
            advantages: Float32Array,
            waitTime: number,
            startTime: number,
            endTime: number,
        },
    ) {
        Promise.all([policyTrainMetrics, valueTrainMetrics]).then(([{ klList, policyLossList }, { valueLossList }]) => {
            for (let i = 0; i < klList.length; i++) {
                const kl = klList[i];

                let policyLoss = 0;
                for (let j = 0; j < miniBatchCount; j++) {
                    policyLoss += policyLossList[i * miniBatchCount + j];
                }

                policyLoss /= miniBatchCount;

                console.log('[Train]: Epoch', i, 'KL:', kl, 'Policy loss:', policyLoss);

                metricsChannels.kl.postMessage(kl);
                metricsChannels.policyLoss.postMessage(kl);
            }

            for (let i = 0; i < CONFIG.epochs; i++) {
                let valueLoss = 0;
                for (let j = 0; j < miniBatchCount; j++) {
                    valueLoss += valueLossList[i * miniBatchCount + j];
                }

                valueLoss /= miniBatchCount;

                console.log('[Train]: Epoch', i, 'Value loss:', valueLoss);

                metricsChannels.valueLoss.postMessage(valueLoss);
            }

            for (const batch of batches) {
                metricsChannels.versionDelta.postMessage(version - batch.version);
                metricsChannels.batchSize.postMessage(batch.size);
            }

            metricsChannels.rewards.postMessage(batches.map(b => b.rewards).flat());
            metricsChannels.values.postMessage(values);
            metricsChannels.returns.postMessage(returns);
            metricsChannels.advantages.postMessage(advantages);
            metricsChannels.waitTime.postMessage(waitTime / 1000);
            metricsChannels.trainTime.postMessage((endTime - startTime) / 1000);
        });

    }
}
