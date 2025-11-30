import Dexie from 'dexie';
import { downloadNetwork } from '../ml/src/Models/Transfer.ts';
import { Model } from '../ml/src/Models/def.ts';
import { forceExitChannel } from './channels.ts';

const db = new Dexie('ui-tank-rl');
db.version(1).stores({ settings: 'key' });

let useNoise = true;
let shouldDraw = false;

// Загружаем настройки из базы (если они там уже сохранены)
async function initSettings() {
    const noiseSetting = await db.table('settings').get('useNoise');
    const drawSetting = await db.table('settings').get('shouldDraw');

    useNoise = noiseSetting ? noiseSetting.value === 'true' : true;
    shouldDraw = drawSetting ? drawSetting.value === 'true' : false;
}

initSettings();

if (globalThis && globalThis.document) {
    document.getElementById('toggleRender')?.addEventListener('click', async () => {
        // Переключаем состояние отрисовки
        shouldDraw = !shouldDraw;
        await db.table('settings').put({ key: 'shouldDraw', value: shouldDraw.toString() });
    });
    document.addEventListener('keypress', async (event) => {
        if (event.code === 'KeyP') {
            shouldDraw = !shouldDraw;
            await db.table('settings').put({ key: 'shouldDraw', value: shouldDraw.toString() });
        }
    });

    document.getElementById('toggleNoise')?.addEventListener('click', async () => {
        useNoise = !useNoise;
        // @ts-ignore
        globalThis.disableNoise = !useNoise;
        await db.table('settings').put({ key: 'useNoise', value: useNoise.toString() });
    });

    document.getElementById('resetState')?.addEventListener('click', async () => {
        // Удаляем сохранённые состояния агента и менеджера из базы настроек
        await db.table('settings').delete('tank-rl-agent-state');
        await db.table('settings').delete('tank-rl-manager-state');
        // Удаляем базу данных tensorflowjs
        localStorage.clear();
        indexedDB.databases().then((dbs) => {
            dbs.forEach((db) => {
                db.name && indexedDB.deleteDatabase(db.name);
            });
            forceExitChannel.postMessage(null);
        });
    });

    document.getElementById('downloadModel')?.addEventListener('click', () => {
        Promise.all([
            downloadNetwork(Model.Policy),
            downloadNetwork(Model.Value),
        ]);
    });
}

export function getDrawState(): boolean {
    return shouldDraw;
}

export function getUseNoise(): boolean {
    return useNoise;
}
