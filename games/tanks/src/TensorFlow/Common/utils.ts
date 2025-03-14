import devtoolsDetect from 'devtools-detect';
import { getSharedAgent } from '../PPO_v2/agent.ts';
import { tanh } from '../../../../../lib/math.ts';

let isVerbose = localStorage.getItem('verbose') === 'true';
let shouldDraw = localStorage.getItem('shouldDraw') === 'true';

document.getElementById('toggleRender')!.addEventListener('click', () => {
    shouldDraw = !shouldDraw;
    localStorage.setItem('shouldDraw', shouldDraw.toString());
});

document.getElementById('toggleVerbose')!.addEventListener('click', () => {
    isVerbose = !isVerbose;
    localStorage.setItem('verbose', isVerbose.toString());
});

document.getElementById('resetState')!.addEventListener('click', () => {
    localStorage.removeItem('tank-rl-agent-state');
    localStorage.removeItem('tank-rl-manager-state');
    indexedDB.deleteDatabase('tensorflowjs');
    window.location.reload();
});

document.getElementById('downloadModel')!.addEventListener('click', () => {
    getSharedAgent().download();
});

export function getDrawState(): boolean {
    return shouldDraw;
}

export function isVerboseLog() {
    return isVerbose && devtoolsDetect.isOpen;
}

export function isDevtoolsOpen() {
    return devtoolsDetect.isOpen;
}

export type Actions = Float32Array;

export function readAction(action: Actions) {
    return {
        shoot: tanh(action[0]) > 0,
        move: tanh(action[1]),
        rotate: tanh(action[2]),
        aimX: tanh(action[3]),
        aimY: tanh(action[4]),
    };
}