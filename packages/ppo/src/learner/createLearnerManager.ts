import { clamp } from 'lodash';
import { catchError, concatMap, EMPTY, first, forkJoin, map, mergeMap, scan, tap } from 'rxjs';
import { exp, log, max } from '../../../../lib/math.ts';
import { bufferWhile } from '../../../../lib/Rx/bufferWhile.ts';
import type { VTraceDiagnostics } from '../metrics/analyzeVTrace.ts';
import { analyzeVTrace } from '../metrics/analyzeVTrace.ts';
import { forceExitChannel, metricsChannels } from '../infra/channels.ts';
import type { PpoConfig } from '../config.ts';
import type { StateBindings } from '../core/StateBindings.ts';
import { flatTypedArray } from '../utils/flat.ts';
import { AgentMemoryBatch, PreparedBatch } from '../memory/Memory.ts';
import { getNetworkSettings } from '../models/networkMeta.ts';
import { Model } from '../models/def.ts';
import { disposeNetwork, getNetwork } from '../models/storage.ts';
import { agentSampleChannel, learnProcessChannel, modelSettingsChannel, queueSizeChannel } from '../core/channels.ts';
import { computeRetraceTargets } from '../core/train.ts';

export type LearnData<S> = PreparedBatch<S> & {
    values: Float32Array,
    returns: Float32Array,
    tdErrors: Float32Array,
    advantages: Float32Array,
};

export function createLearnerManager<S>({ config, bindings, actionHeadDims }: {
    config: PpoConfig,
    bindings: StateBindings<S>,
    actionHeadDims: number[],
}) {
    let lastEndTime = 0;
    let queueSize = 0;

    let rewardRatio = 0;
    let emaStdReturns = 0;

    agentSampleChannel.obs.pipe(
        bufferWhile((batches) => {
            return batches.reduce((acc, b) => acc + b.memoryBatch.size, 0) < config.batchSize(max(...batches.map(s => s.networkVersion), 0));
        }),
        tap(() => queueSizeChannel.emit(queueSize++)),
        concatMap((samples) => {
            const startTime = Date.now();
            const waitTime = lastEndTime === 0 ? undefined : startTime - lastEndTime;

            console.info('Start processing batch', waitTime !== undefined ? `(waited ${waitTime} ms)` : '');

            return forkJoin([
                getNetwork(Model.Policy, config.savePath),
                getNetwork(Model.Value, config.savePath),
            ]).pipe(
                map(([policyNetwork, valueNetwork]): LearnData<S> => {
                    const settings = getNetworkSettings(policyNetwork);
                    rewardRatio = rewardRatio || settings.rewardRatio || 1;
                    emaStdReturns = emaStdReturns || settings.emaStdReturns || 0;

                    const expIteration = settings.expIteration ?? 0;
                    const batchData = squeezeBatches<S>(samples.map(b => {
                        b.memoryBatch.rewards.forEach((_, i, arr) => {
                            arr[i] *= rewardRatio;
                        })
                        return b.memoryBatch as AgentMemoryBatch<S>
                    }));
                    const {pureLogits, ...vTraceBatchData} = computeRetraceTargets<S>(
                        policyNetwork,
                        valueNetwork,
                        batchData,
                        config.miniBatchSize(expIteration),
                        config.gamma(expIteration),
                        actionHeadDims.length,
                        bindings,
                    );
                    const learnData = {
                        ...batchData,
                        ...vTraceBatchData,
                    };

                    metricsChannels.logit.postMessage(pureLogits.map((v, i) => {
                        const step = actionHeadDims[i];
                        const actionIndexes = [];
                        for (let j = 0; j < v.length; j += step) {
                            const logitsSlice = v.subarray(j, j + step);
                            const maxIndex = logitsSlice.reduce((bestIndex, value, index, array) =>
                                value > array[bestIndex] ? index : bestIndex, 0);
                            actionIndexes.push(maxIndex);
                        }
                        return actionIndexes;
                    }));
                    metricsChannels.batchSize.postMessage(samples.map(b => b.memoryBatch.size));
                    metricsChannels.versionDelta.postMessage(samples.map(b => expIteration - b.networkVersion));

                    disposeNetwork(policyNetwork);
                    disposeNetwork(valueNetwork);

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

                            throw new Error(`Model ${envelope.modelName} failed`, {cause: envelope.error});
                        }, {[Model.Policy]: false, [Model.Value]: false}),
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
                            const stdRatio = diagnostics.returns.std > 0
                                ? diagnostics.values.std / diagnostics.returns.std
                                : 0;
                            // compute reward ratio adjustment, for correct scaling of all v-trace components
                            emaStdReturns = (emaStdReturns || diagnostics.returns.std) * 0.9 + diagnostics.returns.std * 0.1;
                            rewardRatio = getRewardRatio(emaStdReturns, rewardRatio);

                            modelSettingsChannel.emit({emaStdReturns, rewardRatio});

                            waitTime !== undefined && metricsChannels.waitTime.postMessage([waitTime / 1000]);
                            metricsChannels.trainTime.postMessage([(lastEndTime - startTime) / 1000]);
                            metricsChannels.vTraceStdRatio.postMessage([stdRatio]);
                            metricsChannels.vTraceExplainedVariance.postMessage([diagnostics.fit.explainedVariance]);

                            reportVTraceAlerts(diagnostics);
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

function squeezeBatches<S>(batches: AgentMemoryBatch<S>[]): PreparedBatch<S> {
    return {
        size: batches.reduce((acc, b) => acc + b.size, 0),
        states: batches.flatMap(b => b.states),
        actions: batches.map(b => b.actions).flat(),
        logits: batches.map(b => b.logits).flat(),
        dones: flatTypedArray(batches.map(b => b.dones)),
        rewards: flatTypedArray(batches.map(b => b.rewards)),
        logProbs: flatTypedArray(batches.map(b => b.logProbs)),
    };
}

function getRewardRatio(stdReturns: number, rewardRatio: number) {
    const target = 1.5; // target std
    const kp = 0.2; // adjustment speed
    const alpha = exp(kp * (log(target) - log(max(stdReturns, 1e-6))));
    const clampedAlpha = clamp(alpha, 0.8, 1.2);
    const nextRewardRatio = rewardRatio * clampedAlpha;

    return nextRewardRatio;
}

function reportVTraceAlerts(diagnostics: VTraceDiagnostics): void {
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
    if (triggeredFlags.length > 0) {
        console.log(`WARNING! VTrace alerts: ${triggeredFlags.join(', ')}`);
    }
}
