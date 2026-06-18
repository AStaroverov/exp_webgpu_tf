/**
 * Debug build world setup: systems-only base game, NO obstacles and NO stand-in
 * driver — the field starts empty so the only entities/actions are the ones the
 * debug GUI spawns/enqueues. Obstacles are added on demand via `spawnObstacles`.
 */

import { createGame, type Game } from "../../engine/src/Game/createGame.ts";
import { GameDI } from "../../engine/src/Game/DI/GameDI.ts";

/** Default debug field: a small, empty 6x6 grid. */
export const DEFAULT_FIELD_SIZE = { cols: 6, rows: 6 };

export async function createDebugGame(
  canvas: HTMLCanvasElement,
  size: { cols: number; rows: number } = DEFAULT_FIELD_SIZE,
): Promise<Game> {
  const game = createGame({
    width: canvas.width,
    height: canvas.height,
    cols: size.cols,
    rows: size.rows,
  });

  await game.setRenderTarget(canvas);

  return game;
}

/** Tear the whole game down and build a fresh field at the given size (empty). */
export async function recreateDebugGame(
  canvas: HTMLCanvasElement,
  size: { cols: number; rows: number } = DEFAULT_FIELD_SIZE,
): Promise<Game> {
  GameDI.destroy();
  return createDebugGame(canvas, size);
}
