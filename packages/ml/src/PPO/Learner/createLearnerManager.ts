import {clamp} from 'lodash';
import {catchError, concatMap, EMPTY, first, forkJoin, map, mergeMap, scan, tap} from 'rxjs';
import {exp, log, max} from '../../../../../lib/math.ts';
import {bufferWhile} from '../../../../../lib/Rx/bufferWhile.ts';
import type {VTraceDiagnostics} from '../../../../ml-common/analyzeVTrace.ts';
import {analyzeVTrace} from '../../../../ml-common/analyzeVTrace.ts';
import {forceExitChannel, metricsChannels} from '../../../../ml-common/channels.ts';
import {CONFIG} from '../../../../ml-common/config.ts';
import {flatTypedArray} from '../../../../ml-common/flat.ts';
import {AgentMemoryBatch} from '../../../../ml-common/Memory.ts';
import {getNetworkSettings} from '../../../../ml-common/utils.ts';
import {Model} from '../../Models/def.ts';
import {disposeNetwork, getNetwork} from '../../Models/Utils.ts';
import {agentSampleChannel, learnProcessChannel, modelSettingsChannel, queueSizeChannel} from '../channels.ts';
import {computeVTraceTargets} from '../train.ts';

export type LearnData = AgentMemoryBatch & {
    values: Float32Array,
    returns: Float32Array,
    tdErrors: Float32Array,
    advantages: Float32Array,

    rewardRatio: number,
    emaStdReturns?: number,
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

            return forkJoin([
                getNetwork(Model.Policy),
                getNetwork(Model.Value),
            ]).pipe(
                map(([policyNetwork, valueNetwork]): LearnData => {
                    const settings = getNetworkSettings(policyNetwork);
                    const expIteration = settings.expIteration ?? 0;
                    const batchData = squeezeBatches(samples.map(b => {
                        b.memoryBatch.rewards.forEach((_, i, arr) => {
                            arr[i] *= settings.rewardRatio ?? 1;
                        })
                        return b.memoryBatch
                    }));
                    const {pureLogStd, pureMean, ...vTraceBatchData} = computeVTraceTargets(
                        policyNetwork,
                        valueNetwork,
                        batchData,
                        CONFIG.miniBatchSize(expIteration),
                        CONFIG.gamma(expIteration),
                        CONFIG.minLogStd(expIteration),
                        CONFIG.maxLogStd(expIteration),
                    );
                    const learnData = {
                        rewardRatio: settings.rewardRatio ?? 1,
                        emaStdReturns: settings.emaStdReturns,
                        ...batchData,
                        ...vTraceBatchData,
                    };

                    metricsChannels.mean.postMessage(pureMean);
                    metricsChannels.logStd.postMessage(pureLogStd);
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
                            const nextEmaStdReturns = (batch.emaStdReturns ?? diagnostics.returns.std) * 0.99 + diagnostics.returns.std * 0.01;
                            const nextRewardRatio = getRewardRatio(nextEmaStdReturns, batch.rewardRatio);

                            modelSettingsChannel.emit({emaStdReturns: nextEmaStdReturns, rewardRatio: nextRewardRatio});

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

function squeezeBatches(batches: AgentMemoryBatch[]): AgentMemoryBatch {
    return {
        size: batches.reduce((acc, b) => acc + b.size, 0),
        states: batches.map(b => b.states).flat(),
        actions: batches.map(b => b.actions).flat(),
        mean: batches.map(b => b.mean).flat(),
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