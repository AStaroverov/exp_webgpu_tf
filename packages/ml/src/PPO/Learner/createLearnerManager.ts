import { pick } from 'lodash-es';
import { RingBuffer } from 'ring-buffer-ts';
import { catchError, concatMap, EMPTY, first, forkJoin, map, mergeMap, scan, tap } from 'rxjs';
import { max } from '../../../../../lib/math.ts';
import { bufferWhile } from '../../../../../lib/Rx/bufferWhile.ts';
import { forceExitChannel, metricsChannels } from '../../../../ml-common/channels.ts';
import { CONFIG } from '../../../../ml-common/config.ts';
import { flatTypedArray } from '../../../../ml-common/flat.ts';
import { AgentMemoryBatch } from '../../../../ml-common/Memory.ts';
import { getNetworkExpIteration } from '../../../../ml-common/utils.ts';
import { Model } from '../../Models/def.ts';
import { disposeNetwork, getNetwork } from '../../Models/Utils.ts';
import {
    CurriculumState,
    curriculumStateChannel,
    EpisodeSample,
    episodeSampleChannel,
    learnProcessChannel,
    queueSizeChannel,
} from '../channels.ts';
import { computeVTraceTargets } from '../train.ts';

export type LearnData = AgentMemoryBatch & {
    values: Float32Array,
    returns: Float32Array,
    tdErrors: Float32Array,
    advantages: Float32Array
};

export function createLearnerManager() {
    let lastEndTime = 0;
    let queueSize = 0;
    const mapScenarioIndexToSuccessRatio = new Map<number, RingBuffer<number>>;
    const mapScenarioIndexToAvgSuccessRatio = new Map<number, number>;
    const computeCurriculumState = (samples: EpisodeSample[]): CurriculumState => {
        for (const sample of samples) {
            if (!mapScenarioIndexToSuccessRatio.has(sample.scenarioIndex)) {
                mapScenarioIndexToSuccessRatio.set(sample.scenarioIndex, new RingBuffer(30));
            }
            mapScenarioIndexToSuccessRatio.get(sample.scenarioIndex)!.add(sample.successRatio);
        }

        for (const [scenarioIndex, successRatioHistory] of mapScenarioIndexToSuccessRatio) {
            const length = successRatioHistory.getBufferLength();
            const size = successRatioHistory.getSize();
            const ratio = length > size / 2
                ? successRatioHistory.toArray().reduce((acc, v) => acc + v, 0) / length
                : 0;

            mapScenarioIndexToAvgSuccessRatio.set(scenarioIndex, ratio);
        }

        return {
            currentVersion: max(...samples.map(s => s.networkVersion), 0),
            mapScenarioIndexToSuccessRatio: Object.fromEntries(mapScenarioIndexToAvgSuccessRatio),
        };
    };

    episodeSampleChannel.obs.pipe(
        bufferWhile((batches) => {
            return batches.reduce((acc, b) => acc + b.memoryBatch.size, 0) < CONFIG.batchSize(max(...batches.map(s => s.networkVersion), 0));
        }),
        tap(() => queueSizeChannel.emit(queueSize++)),
        concatMap((samples) => {
            const startTime = Date.now();
            const waitTime = lastEndTime === 0 ? undefined : startTime - lastEndTime;

            console.info('Start processing batch', waitTime !== undefined ? `(waited ${waitTime} ms)` : '');

            curriculumStateChannel.emit(computeCurriculumState(samples));
            metricsChannels.batchSize.postMessage(samples.map(b => b.memoryBatch.size));
            metricsChannels.successRatio.postMessage(samples.map(b => pick(b, 'scenarioIndex', 'successRatio')));

            return forkJoin([
                getNetwork(Model.Policy),
                getNetwork(Model.Value),
            ]).pipe(
                map(([policyNetwork, valueNetwork]): LearnData => {
                    const version = getNetworkExpIteration(policyNetwork);
                    const batch = squeezeBatches(samples.map(b => b.memoryBatch));
                    const learnData = {
                        ...batch,
                        ...computeVTraceTargets(
                            policyNetwork,
                            valueNetwork,
                            batch,
                            CONFIG.miniBatchSize(version),
                            CONFIG.gamma(version)
                        ),
                    };

                    disposeNetwork(policyNetwork);
                    disposeNetwork(valueNetwork);

                    metricsChannels.versionDelta.postMessage(
                        samples.map(b => version - b.networkVersion),
                    );

                    return learnData;
                }),
                mergeMap((batch) => {
                    return learnProcessChannel.request(batch).pipe(
                        scan((acc, envelope) => {
                            if ('version' in envelope) {
                                acc[envelope.modelName] = true;
                                return acc;
                            }

                            if (envelope.restart) {
                                forceExitChannel.postMessage(null);
                            }

                            throw new Error(`Model ${envelope.modelName} failed`, { cause: envelope.error });
                        }, { [Model.Policy]: false, [Model.Value]: false }),
                        first((state) => state[Model.Policy] && state[Model.Value]),
                        tap(() => {
                            queueSizeChannel.emit(queueSize--);
                            console.info('Batch processed successfully');

                            lastEndTime = Date.now();

                            metricsChannels.rewards.postMessage(batch.rewards);
                            metricsChannels.values.postMessage(batch.values);
                            metricsChannels.returns.postMessage(batch.returns);
                            metricsChannels.tdErrors.postMessage(batch.tdErrors);
                            metricsChannels.advantages.postMessage(batch.advantages);

                            waitTime !== undefined && metricsChannels.waitTime.postMessage([waitTime / 1000]);
                            metricsChannels.trainTime.postMessage([(lastEndTime - startTime) / 1000]);
                        }),
                        catchError((error) => {
                            queueSizeChannel.emit(queueSize--);
                            console.error('Batch processing failed', error);

                            return EMPTY;
                        })
                    )
                }),
            );
        }),
    ).subscribe({
        error: (error) => {
            console.error('Batch processing:', error);
            forceExitChannel.postMessage(null);
        },
    });
}

function squeezeBatches(batches: AgentMemoryBatch[]): AgentMemoryBatch {
    return {
        size: batches.reduce((acc, b) => acc + b.size, 0),
        states: batches.map(b => b.states).flat(),
        actions: batches.map(b => b.actions).flat(),
        mean: batches.map(b => b.mean).flat(),
        logStd: batches.map(b => b.logStd).flat(),
        dones: flatTypedArray(batches.map(b => b.dones)),
        rewards: flatTypedArray(batches.map(b => b.rewards)),
        logProbs: flatTypedArray(batches.map(b => b.logProbs)),
        perturbed: flatTypedArray(batches.map(b => b.perturbed)),
    };
}
