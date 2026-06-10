import { GameDI } from "../unknown/src/Game/DI/GameDI.ts";
import { createDebugGame } from "./src/createDebugGame.ts";
import { createDebugGUI } from "./src/ui/createDebugGUI.ts";
import { createManualControl } from "./src/input/createManualControl.ts";

const canvas = document.getElementById("c") as HTMLCanvasElement;

// Square canvas (CSS is 100vmin square): size the backing store to the same
// square so the render isn't stretched and matches the square play field.
const side = Math.min(window.innerWidth, window.innerHeight) * window.devicePixelRatio;
canvas.width = side;
canvas.height = side;

await createDebugGame(canvas);

const manualControl = createManualControl(canvas);
createDebugGUI(canvas, manualControl);

let prev = performance.now();
const loop = (now: number) => {
  const delta = Math.min(16.6667, now - prev);
  prev = now;
  // Set the manually-controlled tank's controllers *before* gameTick so the
  // physics/turret/bullet systems read them in this same frame.
  manualControl.update(delta);
  // gameTick is briefly null while "Recreate field" tears the game down.
  GameDI.gameTick?.(delta);
  requestAnimationFrame(loop);
};
requestAnimationFrame(loop);
