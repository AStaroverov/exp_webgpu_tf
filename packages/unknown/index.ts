import { createGame } from './src/Game/createGame.ts';

const canvas = document.getElementById('c') as HTMLCanvasElement;
const sndBtn = document.getElementById('snd') as HTMLButtonElement;

canvas.width = window.innerWidth * window.devicePixelRatio;
canvas.height = window.innerHeight * window.devicePixelRatio;

const game = createGame({
    width: canvas.width,
    height: canvas.height,
});

await game.setRenderTarget(canvas);

sndBtn.addEventListener('click', () => {
    game.enableSound();
    sndBtn.remove();
});

let prev = performance.now();
const loop = (now: number) => {
    const delta = Math.min(16.6667, now - prev);
    prev = now;
    game.gameTick(delta);
    requestAnimationFrame(loop);
};
requestAnimationFrame(loop);
