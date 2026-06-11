import Dexie from "dexie";
import { downloadNetwork } from "../../../ppo/src/models/Transfer.ts";
import { Model } from "../../../ppo/src/models/def.ts";
import { forceExitChannel } from "../../../ppo/src/infra/channels.ts";
import { setGreedyInference } from "../env/UnknownAgent.ts";
import { CONFIG } from "../config.ts";

const db = new Dexie("ui-unknown-rl");
db.version(1).stores({ settings: "key" });

// Whether the main tab should run + render a live visual episode. Off by default:
// rendering costs compute the actors could use; charts update regardless.
let shouldDraw = false;
// Whether the visual episode's agents pick actions greedily (argmax) instead of
// sampling. Main-tab only: applying it sets the UnknownAgent override, which
// actor workers never touch.
let greedyInference = false;

async function initSettings() {
  const drawSetting = await db.table("settings").get("shouldDraw");
  shouldDraw = drawSetting ? drawSetting.value === "true" : false;
  const greedySetting = await db.table("settings").get("greedyInference");
  greedyInference = greedySetting ? greedySetting.value === "true" : false;
  setGreedyInference(greedyInference);
}

export const settingsReady = initSettings();

export function getDrawState(): boolean {
  return shouldDraw;
}

export async function setDrawState(value: boolean) {
  shouldDraw = value;
  await db.table("settings").put({ key: "shouldDraw", value: shouldDraw.toString() });
}

export function getGreedyState(): boolean {
  return greedyInference;
}

export async function setGreedyState(value: boolean) {
  greedyInference = value;
  setGreedyInference(value);
  await db.table("settings").put({ key: "greedyInference", value: greedyInference.toString() });
}

export function downloadModels() {
  return Promise.all([
    downloadNetwork(Model.Policy, CONFIG.savePath),
    downloadNetwork(Model.Value, CONFIG.savePath),
  ]);
}

export async function resetState() {
  localStorage.clear();
  const dbs = await indexedDB.databases();
  dbs.forEach((d) => {
    d.name && indexedDB.deleteDatabase(d.name);
  });
  forceExitChannel.postMessage(null);
}
