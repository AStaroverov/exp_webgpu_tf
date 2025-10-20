import { pick } from 'lodash-es';
import { RingBuffer } from 'ring-buffer-ts';
import { catchError, concatMap, EMPTY, first, forkJoin, map, mergeMap, scan, tap } from 'rxjs';
import { max } from '../../../../../lib/math.ts';
import { bufferWhile } from '../../../../../lib/Rx/bufferWhile.ts';
import { forceExitChannel, metricsChannels } from '../../../../ml-common/channels.ts';
import { SAC_CONFIG } from '../../../../ml-common/config.ts';
import { SACMemoryBatch } from '../../../../ml-common/Memory.ts';
import { SACReplayBuffer } from '../../../../ml-common/SACReplayBuffer.ts';
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

// SAC doesn't need advantages/returns - just raw transitions
export type LearnData = SACMemoryBatch;

// Global replay buffer for SAC
const replayBuffer = new SACReplayBuffer(
    SAC_CONFIG.replayBufferSize,
    SAC_CONFIG.prioritizedReplay,
);

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
        // Collect samples and add to replay buffer
        tap((sample) => {
            // Теперь sample.memoryBatch уже правильного типа SACMemoryBatch
            replayBuffer.addBatch(sample.memoryBatch);
            console.info(`Added ${sample.memoryBatch.size} transitions to replay buffer. Total: ${replayBuffer.size()}`);
        }),
        // Buffer episodes until we have enough for a batch
        bufferWhile((batches) => {
            return batches.reduce((acc, b) => acc + b.memoryBatch.size, 0) < SAC_CONFIG.batchSize;
        }),
        tap(() => queueSizeChannel.emit(queueSize++)),
        concatMap((samples) => {
            const startTime = Date.now();
            const waitTime = lastEndTime === 0 ? undefined : startTime - lastEndTime;

            console.info('Start processing batch', waitTime !== undefined ? `(waited ${waitTime} ms)` : '');

            curriculumStateChannel.emit(computeCurriculumState(samples));
            metricsChannels.batchSize.postMessage(samples.map(b => b.memoryBatch.size));
            metricsChannels.successRatio.postMessage(samples.map(b => pick(b, 'scenarioIndex', 'successRatio')));

            // Check if we have enough data in replay buffer
            if (!replayBuffer.canSample(SAC_CONFIG.batchSize)) {
                console.info('Not enough samples in replay buffer yet');
                queueSizeChannel.emit(queueSize--);
                return EMPTY;
            }

            // Sample from replay buffer
            const batch = replayBuffer.sample(SAC_CONFIG.batchSize);
            if (!batch) {
                console.warn('Failed to sample from replay buffer');
                queueSizeChannel.emit(queueSize--);
                return EMPTY;
            }

            return forkJoin([
                getNetwork(Model.Policy),
            ]).pipe(
                map(([policyNetwork]) => {
                    const version = getNetworkExpIteration(policyNetwork);

                    disposeNetwork(policyNetwork);

                    metricsChannels.versionDelta.postMessage(
                        samples.map(b => version - b.networkVersion),
                    );

                    return batch as LearnData;
                }),
                mergeMap((learnBatch) => {
                    return learnProcessChannel.request(learnBatch).pipe(
                        scan((acc, envelope) => {
                            if ('version' in envelope) {
                                // TypeScript workaround for dynamic model names
                                (acc as any)[envelope.modelName] = true;
                                return acc;
                            }

                            if (envelope.restart) {
                                forceExitChannel.postMessage(null);
                            }

                            throw new Error(`Model ${envelope.modelName} failed`, { cause: envelope.error });
                        }, {
                            [Model.Policy]: false,
                            [Model.Critic1]: false,
                            [Model.Critic2]: false
                        } as Record<Model, boolean>),
                        first((state) => state[Model.Policy] && state[Model.Critic1] && state[Model.Critic2]),
                        tap(() => {
                            queueSizeChannel.emit(queueSize--);
                            console.info('Batch processed successfully');

                            lastEndTime = Date.now();

                            metricsChannels.rewards.postMessage(learnBatch.rewards);

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
