import { catchError, concatMap, EMPTY, first, forkJoin, map, mergeMap, scan, tap } from 'rxjs';
import { max } from '../../../../../lib/math.ts';
import { bufferWhile } from '../../../../../lib/Rx/bufferWhile.ts';
import type { VTraceDiagnostics } from '../../../../ml-common/analyzeVTrace.ts';
import { analyzeVTrace } from '../../../../ml-common/analyzeVTrace.ts';
import { forceExitChannel, metricsChannels } from '../../../../ml-common/channels.ts';
import { CONFIG } from '../../../../ml-common/config.ts';
import { flatTypedArray } from '../../../../ml-common/flat.ts';
import { AgentMemoryBatch } from '../../../../ml-common/Memory.ts';
import { getNetworkExpIteration } from '../../../../ml-common/utils.ts';
import { Model } from '../../Models/def.ts';
import { disposeNetwork, getNetwork } from '../../Models/Utils.ts';
import {
    agentSampleChannel,
    learnProcessChannel,
    queueSizeChannel
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

    agentSampleChannel.obs.pipe(
        bufferWhile((batches) => {
            return batches.reduce((acc, b) => acc + b.memoryBatch.size, 0) < CONFIG.batchSize(max(...batches.map(s => s.networkVersion), 0));
        }),
        tap(() => queueSizeChannel.emit(queueSize++)),
        concatMap((samples) => {
            const startTime = Date.now();
            const waitTime = lastEndTime === 0 ? undefined : startTime - lastEndTime;

            console.info('Start processing batch', waitTime !== undefined ? `(waited ${waitTime} ms)` : '');

            metricsChannels.batchSize.postMessage(samples.map(b => b.memoryBatch.size));

            return forkJoin([
                getNetwork(Model.Policy),
                getNetwork(Model.Value),
            ]).pipe(
                map(([policyNetwork, valueNetwork]): LearnData => {
                    const version = getNetworkExpIteration(policyNetwork);
                    const batchData = squeezeBatches(samples.map(b => b.memoryBatch));
                    const { pureLogStd, ...vTraceBatchData } = computeVTraceTargets(
                        policyNetwork,
                        valueNetwork,
                        batchData,
                        CONFIG.miniBatchSize(version),
                        CONFIG.gamma(version),
                        CONFIG.minLogStd(version),
                        CONFIG.maxLogStd(version),
                    );
                    const learnData = {
                        ...batchData,
                        ...vTraceBatchData,
                    };

                    disposeNetwork(policyNetwork);
                    disposeNetwork(valueNetwork);

                    metricsChannels.mean.postMessage(flatTypedArray(batchData.mean));
                    metricsChannels.logStd.postMessage(pureLogStd);
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

                            const diagnostics = analyzeVTrace(
                                batch.advantages,
                                batch.tdErrors,
                                batch.returns,
                                batch.values,
                            );

                            metricsChannels.vTraceExplainedVariance.postMessage([
                                diagnostics.fit.explainedVariance,
                            ]);

                            const stdRatio = diagnostics.returns.std > 0
                                ? diagnostics.values.std / diagnostics.returns.std
                                : 0;
                            metricsChannels.vTraceStdRatio.postMessage([stdRatio]);

                            reportVTraceAlerts(diagnostics, stdRatio);

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

function reportVTraceAlerts(diagnostics: VTraceDiagnostics, stdRatio: number): void {
    const triggeredFlags: string[] = [];

    if (diagnostics.advantages.flags.advStdHigh) {
        triggeredFlags.push('advStdHigh');
    }
    if (diagnostics.advantages.flags.advTailsWide) {
        triggeredFlags.push('advTailsWide');
    }
    if (diagnostics.advantages.flags.advMeanShift) {
        triggeredFlags.push('advMeanShift');
    }
    if (diagnostics.tdErrors.flags.tdMeanShift) {
        triggeredFlags.push('tdMeanShift');
    }
    if (diagnostics.tdErrors.flags.tdHeavyTails) {
        triggeredFlags.push('tdHeavyTails');
    }
    if (diagnostics.fit.scaleMismatch) {
        triggeredFlags.push('scaleMismatch');
    }
    if (diagnostics.fit.explainedVariance < 0.65 || diagnostics.fit.explainedVariance > 0.75) {
        triggeredFlags.push(`explainedVariance=${diagnostics.fit.explainedVariance.toFixed(3)}`);
    }
    if (!Number.isFinite(stdRatio) || stdRatio < 0.8 || stdRatio > 1.2) {
        triggeredFlags.push(`stdRatio=${stdRatio.toFixed(3)}`);
    }

    if (triggeredFlags.length > 0) {
        console.log(`WARNING! VTrace alerts: ${triggeredFlags.join(', ')}`);
    }
}
