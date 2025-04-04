import Dexie from 'dexie';
import devtoolsDetect from 'devtools-detect';

// Инициализируем базу Dexie с таблицей settings
const db = new Dexie('ui-tank-rl');
db.version(1).stores({ settings: 'key' });

// Кэш для хранения значений настроек (аналог localStorage)
let isVerbose = false;
let shouldDraw = false;

// Загружаем настройки из базы (если они там уже сохранены)
async function initSettings() {
    const verboseSetting = await db.table('settings').get('verbose');
    const drawSetting = await db.table('settings').get('shouldDraw');

    isVerbose = verboseSetting ? verboseSetting.value === 'true' : false;
    shouldDraw = drawSetting ? drawSetting.value === 'true' : false;
}

initSettings();

if (globalThis && globalThis.document) {
    document.getElementById('toggleRender')?.addEventListener('click', async () => {
        // Переключаем состояние отрисовки
        shouldDraw = !shouldDraw;
        await db.table('settings').put({ key: 'shouldDraw', value: shouldDraw.toString() });
    });

    document.getElementById('toggleVerbose')?.addEventListener('click', async () => {
        // Переключаем режим подробного логирования
        isVerbose = !isVerbose;
        await db.table('settings').put({ key: 'verbose', value: isVerbose.toString() });
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
            window.location.reload();
        });
    });

    document.getElementById('downloadModel')?.addEventListener('click', () => {
        throw new Error('Not implemented');
    });
}

// Экспортируем функции для получения состояния отрисовки и логирования

export function getDrawState(): boolean {
    return shouldDraw;
}

export function isVerboseLog(): boolean {
    return isVerbose;
}

export function isDevtoolsOpen(): boolean {
    return devtoolsDetect.isOpen;
}
