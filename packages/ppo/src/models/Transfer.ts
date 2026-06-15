import * as tf from "@tensorflow/tfjs";

import "./Layers";

import { isBrowser } from "../../../../lib/detect.ts";
import { onReadyRead } from "../utils/Tensor.ts";
import { getNetworkExpIteration } from "./networkMeta.ts";
import { LAST_NETWORK_VERSION, Model } from "./def.ts";
import {
  getIndexedDBModelPath,
  removeOutLimitNetworks,
  shouldSaveHistoricalVersion,
} from "./storage.ts";

export async function saveNetworkToDB(network: tf.LayersModel, name: Model, savePath: string) {
  await onReadyRead();

  const networkVersion = getNetworkExpIteration(network);
  const shouldSaveHistorical = await shouldSaveHistoricalVersion(name, networkVersion, savePath);

  if (!isBrowser) {
    throw new Error("Unsupported environment for saving model");
  }

  // The "last" network is the one training resumes from — save it first and await it.
  await network.save(getIndexedDBModelPath(name, LAST_NETWORK_VERSION, savePath), {
    includeOptimizer: true,
  });

  if (shouldSaveHistorical) {
    console.info("Saving historical version of network:", name, networkVersion);
    // A checkpoint failure must never crash training: an unhandled rejection here would
    // trigger forceExitChannel -> page reload mid-write. Sequence after the current save
    // (no two concurrent saves of the same model) and swallow errors.
    try {
      await network.save(getIndexedDBModelPath(name, networkVersion, savePath), {
        includeOptimizer: true,
      });
      await removeOutLimitNetworks(name, savePath);
    } catch (error) {
      console.error("Failed to save historical network version:", name, networkVersion, error);
    }
  }
}

export function loadNetworkFromDB(name: Model, version: number, savePath: string) {
  if (isBrowser) {
    return tf.loadLayersModel(getIndexedDBModelPath(name, version, savePath));
  }

  throw new Error("Unsupported environment for loading model");
}

export async function loadLastNetworkFromDB(name: Model, savePath: string) {
  return loadNetworkFromDB(name, LAST_NETWORK_VERSION, savePath);
}

export async function loadNetworkFromFS(path: string, name: Model) {
  if (isBrowser) {
    const modelPath = `${path}/${name}.json`;
    const model = await tf.loadLayersModel(modelPath);

    return model;
  }

  throw new Error("Unsupported environment for loading model");
}

export function downloadNetwork(name: Model, savePath: string) {
  return loadNetworkFromDB(name, LAST_NETWORK_VERSION, savePath).then((network) =>
    network.save(`downloads://${name}`, { includeOptimizer: true }),
  );
}
