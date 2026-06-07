/**
 * Debug build world setup: systems-only base game + obstacles, NO stand-in
 * driver — vehicles stay idle so the only actions are the ones enqueued from
 * the debug GUI.
 */

import { createGame, type Game } from '../../unknown/src/Game/createGame.ts';
import { GameDI } from '../../unknown/src/Game/DI/GameDI.ts';
import { spawnObstacles } from '../../unknown/src/Game/ECS/Entities/Obstacle/spawnObstacles.ts';

export async function createDebugGame(canvas: HTMLCanvasElement): Promise<Game> {
    const game = createGame({
        width: canvas.width,
        height: canvas.height,
    });

    spawnObstacles();

    await game.setRenderTarget(canvas);

    return game;
}

/** Tear the whole game down and build a fresh field (new obstacle layout). */
export async function recreateDebugGame(canvas: HTMLCanvasElement): Promise<Game> {
    GameDI.destroy();
    return createDebugGame(canvas);
}
