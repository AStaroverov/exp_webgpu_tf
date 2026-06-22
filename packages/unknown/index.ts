import { createGame } from "./src/Game/createGame.ts";
import { setupDemoWorld } from "./src/Game/setupDemoWorld.ts";
import { createManualControl } from "./src/Game/input/createManualControl.ts";
import { setTankWeapon } from "./src/Game/ECS/Entities/Tank/setTankWeapon.ts";
import { createWeaponBar } from "./src/ui/createWeaponBar.ts";
import { createHpBar } from "./src/ui/createHpBar.ts";
import { createScoreCounter } from "./src/ui/createScoreCounter.ts";
import { createDeathOverlay } from "./src/ui/createDeathOverlay.ts";
import { getTankHealth } from "./src/Game/ECS/Entities/Tank/TankUtils.ts";
import { GameDI } from "./src/Game/DI/GameDI.ts";
import { getGameComponents } from "./src/Game/ECS/createGameWorld.ts";
import { createPolicyOpponentController } from "../ppo_unknown/src/env/EvalPolicyAgent.ts";
import { MapDI } from "./src/Game/DI/MapDI.ts";
import { SoundManager } from "./src/Game/ECS/Systems/Sound/index.ts";
import { setCameraZoom } from "../renderer/src/ECS/Systems/ResizeSystem.ts";

/** Spawn a fresh policy-driven enemy every this many ms. */
const ENEMY_SPAWN_INTERVAL_MS = 10_000;

// Trained policy weights (tfjs model.json + sidecar bin). `new URL(...,
// import.meta.url)` makes vite emit both as assets and gives their final URLs.
const POLICY_MODEL_URL = new URL("./assets/policy-model.json", import.meta.url).href;
const POLICY_WEIGHTS_URL = new URL("./assets/policy-model.weights.bin", import.meta.url).href;

// CSS pixels — the play field is a rectangular viewport that fits the deck screen.
const CANVAS_W = 760;
const CANVAS_H = 450;
// Hex grid size — a rectangle with roughly the canvas aspect (pointy hexes step
// ~√3·r across, ~1.5·r down, so cols ≈ 1.5·rows fits a ~1.73:1 viewport).
const GRID_COLS = 20;
const GRID_ROWS = 13;
// >1 → zoom past the auto-fit so the grid (and its rock border) spills slightly
// off the screen edges instead of sitting fully inside with padding.
const OVERSCAN = 1.12;

// WebGPU gate: the renderer can't start without it, so detect it up front and
// show a message in place of the game instead of failing deep in GPU init.
async function hasWebGPU(): Promise<boolean> {
  if (!navigator.gpu) return false;
  try {
    return (await navigator.gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

if (!(await hasWebGPU())) {
  document.getElementById("frame")?.remove();
  document.getElementById("snd")?.remove();
  const msg = document.getElementById("no-webgpu");
  if (msg) msg.style.display = "block";
  // Stop module evaluation; the game never starts.
  throw new Error("WebGPU unavailable");
}

const canvas = document.getElementById("c") as HTMLCanvasElement;
const sndBtn = document.getElementById("snd") as HTMLButtonElement;

// CSS size is the world/camera size; the backing store is scaled by DPR for
// crisp rendering. createGame is fed CSS units so camera zoom, the projection
// (ResizeSystem) and the cursor→world unprojection (manual control) all agree.
canvas.style.width = `${CANVAS_W}px`;
canvas.style.height = `${CANVAS_H}px`;
canvas.width = CANVAS_W * window.devicePixelRatio;
canvas.height = CANVAS_H * window.devicePixelRatio;

const game = createGame({
  width: CANVAS_W,
  height: CANVAS_H,
  cols: GRID_COLS,
  rows: GRID_ROWS,
});

// Player (team 1) + the starting enemy (team 2); `spawnEnemy` adds more later.
const { playerEid, enemyEid, spawnEnemy } = setupDemoWorld();

// Zoom in past createGame's auto-fit so the grid overspills the viewport a touch
// (the rock border ends up around / just past the screen edges).
{
  const bounds = MapDI.grid.worldBounds();
  const fit = Math.min(
    CANVAS_W / (bounds.maxX - bounds.minX),
    CANVAS_H / (bounds.maxY - bounds.minY),
  );
  setCameraZoom(fit * OVERSCAN);
}

await game.setRenderTarget(canvas);

// Enemies are driven by the trained policy loaded from assets (one shared driver
// + network). Fire-and-forget: a failed load just leaves enemies idle. The
// starting enemy attaches now; a timer spawns + attaches a fresh one every 20s.
const opponents = createPolicyOpponentController({
  modelJsonUrl: POLICY_MODEL_URL,
  weightsUrl: POLICY_WEIGHTS_URL,
});
const attachEnemy = (eid: number) =>
  opponents.attach(eid).catch((err) => console.error("Failed to attach enemy policy:", err));

attachEnemy(enemyEid);
setInterval(() => {
  const eid = spawnEnemy();
  if (eid) attachEnemy(eid);
}, ENEMY_SPAWN_INTERVAL_MS);

// Keyboard + mouse control of the player's tank (arrows/WASD = hull, mouse =
// turret aim, left button = fire). Updated each frame *before* gameTick so its
// controller writes are consumed this same frame.
const manual = createManualControl(canvas);
manual.setEid(playerEid);
manual.setEnabled(true);

// Weapon selector (player only): four pixel icons on the canvas's bottom edge,
// hotkeys 1–4, swap the gun on the player's turret in place. The tank spawns as
// a MediumTank → "normal".
createWeaponBar(canvas, {
  initial: "normal",
  onSelect: (weapon) => setTankWeapon(playerEid, weapon),
});

// Player HP readout, bottom-left of the canvas; fed each frame from the loop.
const hpBar = createHpBar(canvas);

// Player score readout, top-left of the canvas; fed each frame from the loop.
const scoreCounter = createScoreCounter(canvas);
const { Score } = getGameComponents(GameDI.world);

// Shown when the player dies; Restart fully rebuilds the game via a page reload.
const deathOverlay = createDeathOverlay(canvas, { onRestart: () => location.reload() });
let playerDead = false;

// Sound on/off preference survives the restart (Restart = full page reload).
// Without this every reload drops back to muted until the button is clicked.
const SOUND_PREF_KEY = "sound-enabled";

if (localStorage.getItem(SOUND_PREF_KEY) === "1") {
  // Was on before the reload — re-enable straight away. A freshly reloaded page
  // has no prior user gesture, so the AudioContext may come up suspended; resume
  // it on the first interaction (the buffers decode fine while suspended).
  game.enableSound();
  sndBtn.remove();
  const resumeOnce = () => {
    SoundManager.resume();
    window.removeEventListener("pointerdown", resumeOnce);
    window.removeEventListener("keydown", resumeOnce);
  };
  window.addEventListener("pointerdown", resumeOnce);
  window.addEventListener("keydown", resumeOnce);
} else {
  sndBtn.addEventListener("click", () => {
    game.enableSound();
    localStorage.setItem(SOUND_PREF_KEY, "1");
    sndBtn.remove();
  });
}

let prev = performance.now();
const loop = (now: number) => {
  const delta = Math.min(16.6667, now - prev);
  prev = now;
  manual.update(delta);
  game.gameTick(delta);

  const health = getTankHealth(playerEid);
  hpBar.setHealth(health);
  scoreCounter.setScore(Score.get(playerEid));
  if (!playerDead && health <= 0) {
    playerDead = true;
    deathOverlay.show();
  }
  requestAnimationFrame(loop);
};
requestAnimationFrame(loop);
