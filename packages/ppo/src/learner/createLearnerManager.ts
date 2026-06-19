import { catchError, concatMap, EMPTY, first, forkJoin, map, mergeMap, scan, tap } from "rxjs";
import { max } from "../../../../lib/math.ts";
import { bufferWhile } from "../../../../lib/Rx/bufferWhile.ts";
import type { VTraceDiagnostics } from "../metrics/analyzeVTrace.ts";
import { analyzeVTrace } from "../metrics/analyzeVTrace.ts";
import { forceExitChannel, metricsChannels } from "../infra/channels.ts";
import type * as tf from "@tensorflow/tfjs";
import type { PpoConfig } from "../config.ts";
import { flatTypedArray } from "../utils/flat.ts";
import { AgentMemoryBatch, PreparedBatch } from "../memory/Memory.ts";
import { getNetworkSettings } from "../models/networkMeta.ts";
import { Model } from "../models/def.ts";
import { disposeNetwork, getNetwork } from "../models/storage.ts";
import { agentSampleChannel, learnProcessChannel, queueSizeChannel } from "../core/channels.ts";
import { computeRetraceTargets } from "../core/train.ts";

export type LearnData<S> = PreparedBatch<S> & {
  values: Float32Array;
  returns: Float32Array;
  tdErrors: Float32Array;
  advantages: Float32Array;
};

export function createLearnerManager<S>({
  config,
  createInputTensors,
  actionHeadDims,
}: {
  config: PpoConfig;
  createInputTensors: (batch: S[]) => tf.Tensor[];
  actionHeadDims: number[];
}) {
  let lastEndTime = 0;
  let queueSize = 0;

  agentSampleChannel.obs
    .pipe(
      bufferWhile((batches) => {
        return (
          batches.reduce((acc, b) => acc + b.memoryBatch.size, 0) <
          config.batchSize(max(...batches.map((s) => s.networkVersion), 0))
        );
      }),
      tap(() => queueSizeChannel.emit(queueSize++)),
      concatMap((samples) => {
        const startTime = Date.now();
        const waitTime = lastEndTime === 0 ? undefined : startTime - lastEndTime;

        console.info(
          "Start processing batch",
          waitTime !== undefined ? `(waited ${waitTime} ms)` : "",
        );

        return forkJoin([
          getNetwork(Model.Policy, config.savePath),
          getNetwork(Model.Value, config.savePath),
        ]).pipe(
          map(([policyNetwork, valueNetwork]): LearnData<S> => {
            const settings = getNetworkSettings(policyNetwork);
            const expIteration = settings.expIteration ?? 0;
            const batchData = squeezeBatches<S>(
              samples.map((b) => b.memoryBatch as AgentMemoryBatch<S>),
            );
            const { pureLogits, ...vTraceBatchData } = computeRetraceTargets<S>(
              policyNetwork,
              valueNetwork,
              createInputTensors,
              batchData,
              config.miniBatchSize(expIteration),
              config.gamma(expIteration),
              actionHeadDims.length,
              undefined,
              batchData.masks,
            );
            const learnData = {
              ...batchData,
              ...vTraceBatchData,
            };

            // Per head: per logit index → per-sample values across the batch.
            metricsChannels.logit.postMessage(
              pureLogits.map((v, i) => {
                const step = actionHeadDims[i];
                const series: number[][] = Array.from({ length: step }, () => []);
                for (let j = 0; j < v.length; j++) {
                  series[j % step].push(v[j]);
                }
                return series;
              }),
            );
            metricsChannels.batchSize.postMessage(samples.map((b) => b.memoryBatch.size));
            metricsChannels.versionDelta.postMessage(
              samples.map((b) => expIteration - b.networkVersion),
            );
            metricsChannels.version.postMessage([expIteration]);

            disposeNetwork(policyNetwork);
            disposeNetwork(valueNetwork);

            return learnData;
          }),
          mergeMap((batch) => {
            return learnProcessChannel.request(batch).pipe(
              scan(
                (acc, envelope) => {
                  if ("version" in envelope) {
                    acc[envelope.modelName] = true;
                    return acc;
                  }

                  if (envelope.restart) {
                    forceExitChannel.postMessage(null);
                  }

                  throw new Error(`Model ${envelope.modelName} failed`, { cause: envelope.error });
                },
                { [Model.Policy]: false, [Model.Value]: false },
              ),
              first((state) => state[Model.Policy] && state[Model.Value]),
              tap(() => {
                queueSizeChannel.emit(queueSize--);
                console.info("Batch processed successfully");

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
                const stdRatio =
                  diagnostics.returns.std > 0
                    ? diagnostics.values.std / diagnostics.returns.std
                    : 0;

                if (waitTime !== undefined) metricsChannels.waitTime.postMessage([waitTime / 1000]);
                metricsChannels.trainTime.postMessage([(lastEndTime - startTime) / 1000]);
                metricsChannels.vTraceStdRatio.postMessage([stdRatio]);
                metricsChannels.vTraceExplainedVariance.postMessage([
                  diagnostics.fit.explainedVariance,
                ]);

                reportVTraceAlerts(diagnostics);
              }),
              catchError((error) => {
                queueSizeChannel.emit(queueSize--);
                console.error("Batch processing failed", error);

                return EMPTY;
              }),
            );
          }),
        );
      }),
    )
    .subscribe({
      error: (error) => {
        console.error("Batch processing:", error);
        forceExitChannel.postMessage(null);
      },
    });
}

function squeezeBatches<S>(batches: AgentMemoryBatch<S>[]): PreparedBatch<S> {
  return {
    size: batches.reduce((acc, b) => acc + b.size, 0),
    states: batches.flatMap((b) => b.states),
    actions: batches.map((b) => b.actions).flat(),
    logits: batches.map((b) => b.logits).flat(),
    masks: batches.some((b) => b.masks) ? batches.flatMap((b) => b.masks ?? []) : undefined,
    dones: flatTypedArray(batches.map((b) => b.dones)),
    rewards: flatTypedArray(batches.map((b) => b.rewards)),
    logProbs: flatTypedArray(batches.map((b) => b.logProbs)),
  };
}

function reportVTraceAlerts(diagnostics: VTraceDiagnostics): void {
  const triggeredFlags: string[] = [];

  if (diagnostics.advantages.flags.advStdHigh) {
    triggeredFlags.push("advStdHigh");
  }
  if (diagnostics.advantages.flags.advTailsWide) {
    triggeredFlags.push("advTailsWide");
  }
  if (diagnostics.advantages.flags.advMeanShift) {
    triggeredFlags.push("advMeanShift");
  }
  if (diagnostics.tdErrors.flags.tdMeanShift) {
    triggeredFlags.push("tdMeanShift");
  }
  if (diagnostics.tdErrors.flags.tdHeavyTails) {
    triggeredFlags.push("tdHeavyTails");
  }
  if (diagnostics.fit.scaleMismatch) {
    triggeredFlags.push("scaleMismatch");
  }
  if (triggeredFlags.length > 0) {
    console.log(`WARNING! VTrace alerts: ${triggeredFlags.join(", ")}`);
  }
}
