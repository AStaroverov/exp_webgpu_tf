import { forceExitChannel, metricsChannels } from '../../Common/channels.ts';
import { actorMemoryChannel, learnMemoryChannel, queueSizeChannel } from '../channels.ts';
import { concatMap, first, forkJoin, map, mergeMap, scan, tap } from 'rxjs';
import { Model } from '../../Models/Transfer.ts';
import { flatTypedArray } from '../../Common/flat.ts';
import { Batch } from '../../Common/Memory.ts';
import { computeVTraceTargets } from '../train.ts';
import { bufferWhile } from '../../../../../../lib/Rx/bufferWhile.ts';
import { CONFIG } from '../config.ts';
import { getNetworkVersion } from '../../Common/utils.ts';
import { disposeNetwork, getNetwork } from '../../Models/Utils.ts';

export type LearnBatch = Batch & {
    values: Float32Array,
    returns: Float32Array,
    tdErrors: Float32Array,
    advantages: Float32Array
};

export function createLearnerManager() {
    let lastBufferTime = 0;
    let queueSize = 0;

    actorMemoryChannel.obs.pipe(
        bufferWhile((batches) => batches.reduce((acc, b) => acc + b.memories.size, 0) < CONFIG.batchSize),
        tap(() => queueSizeChannel.emit(queueSize++)),
        concatMap((batches) => {
            console.info('Start processing batch');
            const startTime = Date.now();
            const waitTime = lastBufferTime === 0 ? 0 : startTime - lastBufferTime;
            lastBufferTime = startTime;

            return forkJoin([
                getNetwork(Model.Policy),
                getNetwork(Model.Value),
            ]).pipe(
                map(([policyNetwork, valueNetwork]): LearnBatch => {
                    const batch = squeezeBatches(batches.map(b => b.memories));
                    const learnBatch = {
                        ...batch,
                        ...computeVTraceTargets(policyNetwork, valueNetwork, batch),
                    };

                    disposeNetwork(policyNetwork);
                    disposeNetwork(valueNetwork);

                    metricsChannels.versionDelta.postMessage(
                        batches.map(b => getNetworkVersion(policyNetwork) - b.version),
                    );
                    metricsChannels.batchSize.postMessage(
                        batches.map(b => b.memories.size),
                    );

                    return learnBatch;
                }),
                mergeMap((batch) => {
                    return learnMemoryChannel.request(batch).pipe(
                        scan((acc, envelope) => {
                            if ('version' in envelope) {
                                acc[envelope.modelName] = true;
                                return acc;
                            }

                            throw new Error(`Model ${ envelope.modelName } error: ${ envelope.error }`);
                        }, { [Model.Policy]: false, [Model.Value]: false }),
                        first((state) => state[Model.Policy] && state[Model.Value]),
                        tap(() => {
                            queueSize--;
                            console.info('Batch processed successfully');

                            const endTime = Date.now();

                            metricsChannels.rewards.postMessage(batch.rewards);
                            metricsChannels.values.postMessage(batch.values);
                            metricsChannels.returns.postMessage(batch.returns);
                            metricsChannels.tdErrors.postMessage(batch.tdErrors);
                            metricsChannels.advantages.postMessage(batch.advantages);

                            waitTime > 0 && metricsChannels.waitTime.postMessage(waitTime / 1000);
                            metricsChannels.trainTime.postMessage((endTime - startTime) / 1000);
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

function squeezeBatches(batches: Batch[]): Batch {
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
