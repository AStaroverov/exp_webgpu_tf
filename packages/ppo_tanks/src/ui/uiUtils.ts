import Dexie from 'dexie';
import { downloadNetwork } from '../../../ppo/src/models/Transfer.ts';
import { Model } from '../../../ppo/src/models/def.ts';
import { forceExitChannel } from '../../../ppo/src/infra/channels.ts';
import { CONFIG } from '../config.ts';

const db = new Dexie('ui-tank-rl');
db.version(1).stores({ settings: 'key' });

let useNoise = true;
let shouldDraw = false;

async function initSettings() {
    const noiseSetting = await db.table('settings').get('useNoise');
    const drawSetting = await db.table('settings').get('shouldDraw');

    useNoise = noiseSetting ? noiseSetting.value === 'true' : true;
    shouldDraw = drawSetting ? drawSetting.value === 'true' : false;
}

export const settingsReady = initSettings();

export function getDrawState(): boolean {
    return shouldDraw;
}

export async function setDrawState(value: boolean) {
    shouldDraw = value;
    await db.table('settings').put({ key: 'shouldDraw', value: shouldDraw.toString() });
}

export function getUseNoise(): boolean {
    return useNoise;
}

export async function setUseNoise(value: boolean) {
    useNoise = value;
    // @ts-ignore
    globalThis.disableNoise = !useNoise;
    await db.table('settings').put({ key: 'useNoise', value: useNoise.toString() });
}

export async function resetState() {
    await db.table('settings').delete('tank-rl-agent-state');
    await db.table('settings').delete('tank-rl-manager-state');
    localStorage.clear();
    const dbs = await indexedDB.databases();
    dbs.forEach((d) => {
        d.name && indexedDB.deleteDatabase(d.name);
    });
    forceExitChannel.postMessage(null);
}

export function downloadModels() {
    return Promise.all([
        downloadNetwork(Model.Policy, CONFIG.savePath),
        downloadNetwork(Model.Value, CONFIG.savePath),
    ]);
}
