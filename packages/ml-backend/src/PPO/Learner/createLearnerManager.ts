import { createRequire } from 'module';
import { catchError, concatMap, EMPTY, first, forkJoin, from, map, merge, mergeMap, scan, tap } from 'rxjs';
import { max } from '../../../../../lib/math.ts';
import { bufferWhile } from '../../../../../lib/Rx/bufferWhile.ts';
import { flatTypedArray } from '../../../../ml-common/flat.ts';
import { AgentMemoryBatch } from '../../../../ml-common/Memory.ts';
import { getNetworkExpIteration } from '../../../../ml-common/utils.ts';
import { Model } from '../../Models/def.ts';
import { deleteExperienceBatch, downloadCurriculumState, downloadExperienceBatch, downloadRecentExperienceBatches, uploadCurriculumState } from '../../Models/supabaseStorage.ts';
import { getNetwork } from '../../Models/Utils.ts';
import { forceExitChannel } from '../../Utils/channels.ts'; // metrics disabled
import { CONFIG } from '../config.ts';
import {
    CurriculumState,
    EpisodeSample,
    episodeSampleChannel,
    queueSizeChannel,
} from '../globalChannels.ts';
import {
    learnProcessChannel,
} from '../localChannels.ts';
import { computeVTraceTargets } from '../train.ts';

const require = createRequire(import.meta.url);
const RingBufferModule = require('ring-buffer-ts');

const RingBuffer = RingBufferModule.RingBuffer;

export type LearnData = AgentMemoryBatch & {
    values: Float32Array,
    returns: Float32Array,
    tdErrors: Float32Array,
    advantages: Float32Array
};

export function createLearnerManager() {
    let lastEndTime = 0;
    let queueSize = 0;
    // Curriculum disabled (no client-side scenarios)
    const lastCurriculumStatePromise = downloadCurriculumState();
    const mapScenarioIndexToSuccessRatio = new Map<number, typeof RingBuffer>();
    const mapScenarioIndexToAvgSuccessRatio = new Map<number, number>();
    const computeCurriculumState = async (samples: EpisodeSample[]): Promise<CurriculumState> => {
        const lastCurriculumState = await lastCurriculumStatePromise;

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
                ? successRatioHistory.toArray().reduce((acc: number, v: number) => acc + v, 0) / length
                : 0;

            mapScenarioIndexToAvgSuccessRatio.set(scenarioIndex, ratio);
        }

        return {
            currentVersion: max(lastCurriculumState.currentVersion, ...samples.map(s => s.networkVersion), 0),
            mapScenarioIndexToSuccessRatio: {
                ...lastCurriculumState.mapScenarioIndexToSuccessRatio,
                ...Object.fromEntries(mapScenarioIndexToAvgSuccessRatio),
            }
        };
    };

    const newSamples = episodeSampleChannel.obs.pipe(
        mergeMap((batchInfo) => {
            console.info(`📥 Downloading batch: ${batchInfo.batchId}`);
            return from(downloadExperienceBatch(batchInfo.fileName)).pipe(
                catchError(() => EMPTY)
            )
        }),
    )
    const recentSamples = from(downloadRecentExperienceBatches(40)).pipe(
        mergeMap(batches => from(batches)),
    );

    merge(newSamples, recentSamples).pipe(
        // merge(newSamples).pipe(
        bufferWhile((batches) => {
            const size = batches.reduce((acc, b) => acc + b.memoryBatch.size, 0);
            const requiredSize = CONFIG.batchSize(max(...batches.map(s => s.networkVersion), 0));
            console.log(`Current buffered size: ${size}, required size: ${requiredSize}`);

            return size < requiredSize;
        }),
        tap(() => queueSizeChannel.emit(queueSize++)),
        concatMap((samples) => {
            const startTime = Date.now();
            const waitTime = lastEndTime === 0 ? undefined : startTime - lastEndTime;

            console.info('Start processing batch', waitTime !== undefined ? `(waited ${waitTime} ms)` : '');

            computeCurriculumState(samples).then((curriculumState) => {
                uploadCurriculumState(curriculumState).catch((error) => {
                    console.error('Failed to upload curriculum state:', error);
                });
            });
            // metrics disabled: batchSize, successRatio

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

                    policyNetwork.dispose();
                    valueNetwork.dispose();

                    // metricsChannels.versionDelta.postMessage(
                    //     samples.map(b => version - b.networkVersion),
                    // ); // disabled metrics

                    return learnData;
                }),
                mergeMap((batch) => {
                    return learnProcessChannel.request(batch).pipe(
                        scan((acc, envelope) => {
                            if ('version' in envelope) {
                                acc[envelope.modelName] = true;
                                return acc;
                            }

                            throw new Error(`Model ${envelope.modelName} error: ${envelope.error}`);
                        }, { [Model.Policy]: false, [Model.Value]: false }),
                        first((state) => state[Model.Policy] && state[Model.Value]),
                        tap(async () => {
                            lastEndTime = Date.now();
                            queueSizeChannel.emit(queueSize--);

                            console.info('Batch processed successfully', ((lastEndTime - startTime) / 1000) + 's');

                            // Delete processed batches from Supabase
                            deleteExperienceBatch(samples.map(s => `${s.batchId}.json`));


                            // metricsChannels.rewards.postMessage(batch.rewards);
                            // metricsChannels.values.postMessage(batch.values);
                            // metricsChannels.returns.postMessage(batch.returns);
                            // metricsChannels.tdErrors.postMessage(batch.tdErrors);
                            // metricsChannels.advantages.postMessage(batch.advantages);

                            // waitTime !== undefined && metricsChannels.waitTime.postMessage([waitTime / 1000]);
                            // metricsChannels.trainTime.postMessage([(lastEndTime - startTime) / 1000]);
                        }),
                    );
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
    };
}
