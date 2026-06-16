import * as tf from "@tensorflow/tfjs";
import { LayersModel } from "@tensorflow/tfjs";
import { isBrowser } from "../../../../lib/detect.ts";
import { random, randomRangeInt } from "../../../../lib/random.ts";
import { patientAction } from "../utils/patientAction.ts";
import { LAST_NETWORK_VERSION, Model, NetworkInfo } from "./def.ts";
import { loadLastNetworkFromDB, loadNetworkFromDB } from "./Transfer.ts";

export function disposeNetwork(network?: LayersModel) {
  network?.optimizer?.dispose();
  network?.dispose();
}

export function getStorePath(name: string, version: number, savePath: string): string {
  const postfix = version === LAST_NETWORK_VERSION ? "" : `|version:${version}`;
  return `${savePath}-${name}${postfix}`;
}

export function getVersionFromStorePath(path: string): number {
  const splitted = path.split("|version:");
  return splitted.length == 2 ? Number(splitted[1]) : LAST_NETWORK_VERSION;
}

export function getIndexedDBModelPath(name: string, version: number, savePath: string): string {
  return `indexeddb://${getStorePath(name, version, savePath)}`;
}

export async function getNetwork(
  modelName: Model,
  savePath: string,
  _getInitial?: () => tf.LayersModel,
) {
  let network: undefined | tf.LayersModel;

  try {
    // Always retry the DB load before falling back to a fresh network. A transient
    // IndexedDB failure (e.g. right after a page reload, mid-write) must not cause
    // getInitial() to overwrite a good saved model with random weights.
    network = await patientAction(() => loadLastNetworkFromDB(modelName, savePath), 10);
  } catch (error) {
    console.warn(`[getNetwork] Could not load model ${modelName} from DB:`, error);
    throw error;
    // network = _getInitial?.();
  }

  if (!network) {
    throw new Error(`Failed to load model ${modelName}`);
  }

  return network;
}

export async function getRandomHistoricalNetwork(modelName: Model, savePath: string) {
  const randomInfo = await getRandomNetworkInfo(modelName, savePath);
  let version = getVersionFromStorePath(randomInfo.name);
  version = Number.isNaN(version) ? LAST_NETWORK_VERSION : version;
  return patientAction(() => loadNetworkFromDB(modelName, version, savePath), 10);
}

export async function getNetworkInfoList(model: Model, savePath: string) {
  const defaultSubName = getStorePath(model, LAST_NETWORK_VERSION, savePath);
  return tf.io.listModels().then((v) =>
    Object.entries(v).reduce((acc, [name, info]) => {
      if (name.includes(defaultSubName)) {
        acc.push({
          name: name.split("://")[1],
          path: name,
          dateSaved: info.dateSaved,
        });
      }
      return acc;
    }, [] as NetworkInfo[]),
  );
}

export async function getRandomNetworkInfo(model: Model, savePath: string) {
  const list = await getNetworkInfoList(model, savePath);
  return list[randomRangeInt(0, list.length - 1)];
}

export async function getPenultimateNetworkVersion(
  name: Model,
  savePath: string,
): Promise<number | undefined> {
  const defaultSubName = getStorePath(name, LAST_NETWORK_VERSION, savePath);
  const penultimate = (await getNetworkInfoList(name, savePath)).reduce(
    (acc, info) => {
      if (info.name !== defaultSubName && (acc === undefined || acc.dateSaved < info.dateSaved)) {
        return info;
      }
      return acc;
    },
    undefined as undefined | NetworkInfo,
  );

  return penultimate === undefined ? undefined : getVersionFromStorePath(penultimate.name);
}

export async function shouldSaveHistoricalVersion(name: Model, version: number, savePath: string) {
  const step = 200_000;
  const penultimateNetworkVersion = await getPenultimateNetworkVersion(name, savePath);

  if (penultimateNetworkVersion == null) {
    return version > step;
  }

  if (Number.isNaN(penultimateNetworkVersion)) {
    console.error("penultimateNetworkVersion is NaN");
    return random() > 0.9;
  }

  return version - penultimateNetworkVersion > step;
}

const NETWORKS_LIMIT_COUNT = 20;

export async function removeOutLimitNetworks(name: Model, savePath: string) {
  const list = await getNetworkInfoList(name, savePath);

  if (list.length > NETWORKS_LIMIT_COUNT) {
    const forRemove = list
      .sort((a, b) => a.dateSaved.getTime() - b.dateSaved.getTime())
      .slice(0, list.length - NETWORKS_LIMIT_COUNT);

    if (isBrowser) {
      for (const networkInfo of forRemove) {
        tf.io.removeModel(networkInfo.path);
      }
    } else {
      throw new Error("Unsupported environment for removing model");
    }
  }
}
