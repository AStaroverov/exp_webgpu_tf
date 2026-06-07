import { GameDI } from '../unknown/src/Game/DI/GameDI.ts';
import { createDebugGame } from './src/createDebugGame.ts';
import { createDebugGUI } from './src/ui/createDebugGUI.ts';

const canvas = document.getElementById('c') as HTMLCanvasElement;

canvas.width = window.innerWidth * window.devicePixelRatio;
canvas.height = window.innerHeight * window.devicePixelRatio;

await createDebugGame(canvas);

createDebugGUI(canvas);

let prev = performance.now();
const loop = (now: number) => {
    const delta = Math.min(16.6667, now - prev);
    prev = now;
    // gameTick is briefly null while "Recreate field" tears the game down.
    GameDI.gameTick?.(delta);
    requestAnimationFrame(loop);
};
requestAnimationFrame(loop);
