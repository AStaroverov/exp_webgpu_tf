import devtoolsDetect from 'devtools-detect';

// Кэш для хранения значений настроек
let isVerbose = false;
let shouldDraw = false;

// Ключи в localStorage
const VERBOSE_KEY = 'verbose';
const DRAW_KEY = 'shouldDraw';
const AGENT_STATE_KEY = 'tank-rl-agent-state';
const MANAGER_STATE_KEY = 'tank-rl-manager-state';

// Инициализация настроек из localStorage
function initSettings() {
    const verboseSetting = localStorage.getItem(VERBOSE_KEY);
    const drawSetting = localStorage.getItem(DRAW_KEY);

    isVerbose = verboseSetting === 'true';
    shouldDraw = drawSetting === 'true';
}

initSettings();

if (globalThis && globalThis.document) {
    globalThis.document.getElementById('toggleRender')?.addEventListener('click', () => {
        shouldDraw = !shouldDraw;
        localStorage.setItem(DRAW_KEY, shouldDraw.toString());
    });

    globalThis.document.getElementById('toggleVerbose')?.addEventListener('click', () => {
        isVerbose = !isVerbose;
        localStorage.setItem(VERBOSE_KEY, isVerbose.toString());
    });

    globalThis.document.getElementById('resetState')?.addEventListener('click', async () => {
        localStorage.removeItem(AGENT_STATE_KEY);
        localStorage.removeItem(MANAGER_STATE_KEY);
        localStorage.removeItem(VERBOSE_KEY);
        localStorage.removeItem(DRAW_KEY);

        const dbs = await indexedDB.databases();
        dbs.forEach((db) => {
            db.name && indexedDB.deleteDatabase(db.name);
        });

        window.location.reload();
    });

    globalThis.document.getElementById('downloadModel')?.addEventListener('click', () => {
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
