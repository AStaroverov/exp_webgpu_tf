import * as tf from "@tensorflow/tfjs";
import { get, isNumber } from "lodash-es";
import type { PpoConfig } from "../config.ts";
import { getNetworkExpIteration, setNetworkSettings } from "../models/networkMeta.ts";
import { patientAction } from "../utils/patientAction.ts";
import { saveNetworkToDB } from "../models/Transfer.ts";
import { disposeNetwork, getNetwork } from "../models/storage.ts";
import { Model } from "../models/def.ts";
import { learnProcessChannel, modelSettingsChannel } from "../core/channels.ts";
import { networkHealthCheck } from "../core/train.ts";
import { LearnData } from "./createLearnerManager.ts";

export async function createLearnerAgent<S>({
  config,
  createInputTensors,
  prepareRandomInputArrays,
  modelName,
  createNetwork,
  trainNetwork,
  onNetworkReady,
}: {
  config: PpoConfig;
  createInputTensors: (batch: S[]) => tf.Tensor[];
  prepareRandomInputArrays: () => S;
  modelName: Model;
  createNetwork: () => tf.LayersModel;
  trainNetwork: (network: tf.LayersModel, batch: LearnData<S>) => unknown | Promise<unknown>;
  onNetworkReady?: (network: tf.LayersModel) => void;
}) {
  let network = await getNetwork(modelName, config.savePath, () => {
    const newNetwork = createNetwork();
    patientAction(() => saveNetworkToDB(newNetwork, modelName, config.savePath));
    return newNetwork;
  });

  modelSettingsChannel.obs.subscribe((settings) => {
    if (isNumber(settings.lr)) {
      settings.lr = settings.lr * (modelName === Model.Value ? config.valueLRCoeff : 1);
    }
    setNetworkSettings(network, settings);
  });

  learnProcessChannel.response(async (rawBatch) => {
    const batch = rawBatch as LearnData<S>;
    try {
      await trainNetwork(network, batch);
      await patientAction(() =>
        networkHealthCheck(network, createInputTensors, prepareRandomInputArrays),
      );
      await patientAction(() => saveNetworkToDB(network, modelName, config.savePath));

      return { modelName: modelName, version: getNetworkExpIteration(network) };
    } catch (e: Error | unknown) {
      console.error(e);

      disposeNetwork(network);
      console.info("Load last network after error...");
      network = await patientAction(() => getNetwork(modelName, config.savePath));

      return {
        modelName: modelName,
        error: get(e, "message") ?? "Unknown error",
        restart: e instanceof Error && e.message.includes("mem"),
      };
    }
  });

  onNetworkReady?.(network);
}
