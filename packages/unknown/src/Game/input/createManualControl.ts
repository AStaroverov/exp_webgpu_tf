/**
 * Manual keyboard + mouse control of a single tank.
 *
 *   - Arrow keys (or WASD) drive the hull: Up/Down = forward/back, Left/Right = turn.
 *   - The mouse aims the turret (it tracks the cursor's world position).
 *   - Holding the left mouse button fires (auto-repeats, rate-limited by reload).
 *
 * It bypasses the action queue and writes the same controller components the
 * action executors do (`VehicleController` for the hull, `TurretController` for
 * the turret), so manual input must NOT be mixed with queued actions on the same
 * tank — a queued MoveStep/Aim/Fire runs inside `gameTick` and would overwrite
 * what we set here (the `PlayerControlled` marker keeps the AI driver off it).
 * `update()` is therefore called once per frame *before* `gameTick`, so the
 * values it writes are consumed by the physics/turret/bullet systems in the very
 * same frame.
 */

import { hasComponent } from "bitecs";
import { GameDI } from "../DI/GameDI.ts";
import { getGameComponents } from "../ECS/createGameWorld.ts";
import { normalizeAngle } from "../../../../../lib/math.ts";
import { cameraPosition, cameraZoom } from "../../../../renderer/src/ECS/Systems/ResizeSystem.ts";

/** Heading-error band (rad) below which the turret steers proportionally — mirrors AimAction. */
const SLOW_BAND = 0.3;
/** Stop nudging the turret once it is this close (rad) to avoid jitter. */
const AIM_TOLERANCE = 0.02;

export type ManualControl = {
  /** The tank entity under manual control (0 = none). */
  setEid(eid: number): void;
  /** Toggle the whole controller on/off. */
  setEnabled(enabled: boolean): void;
  /** Per-frame: read input, write the controllers. Call before `gameTick`. */
  update(delta: number): void;
  /** Detach the DOM listeners. */
  dispose(): void;
};

export function createManualControl(canvas: HTMLCanvasElement): ManualControl {
  let eid = 0;
  let enabled = false;

  // Pressed state of the four directional keys.
  const keys = { up: false, down: false, left: false, right: false };
  // Mouse cursor in CSS pixels relative to the canvas, and left-button state.
  const mouse = { x: 0, y: 0, inside: false, down: false };

  // Arrows and WASD both drive — whichever the player reaches for.
  const KEY_MAP: Record<string, keyof typeof keys> = {
    ArrowUp: "up",
    ArrowDown: "down",
    ArrowLeft: "left",
    ArrowRight: "right",
    w: "up",
    s: "down",
    a: "left",
    d: "right",
    W: "up",
    S: "down",
    A: "left",
    D: "right",
  };

  const onKeyDown = (e: KeyboardEvent) => {
    const k = KEY_MAP[e.key];
    if (k === undefined) return;
    keys[k] = true;
    e.preventDefault(); // arrows would otherwise scroll the page
  };
  const onKeyUp = (e: KeyboardEvent) => {
    const k = KEY_MAP[e.key];
    if (k === undefined) return;
    keys[k] = false;
    e.preventDefault();
  };
  const onMouseMove = (e: MouseEvent) => {
    const rect = canvas.getBoundingClientRect();
    mouse.x = e.clientX - rect.left;
    mouse.y = e.clientY - rect.top;
    mouse.inside = mouse.x >= 0 && mouse.y >= 0 && mouse.x <= rect.width && mouse.y <= rect.height;
  };
  const onMouseDown = (e: MouseEvent) => {
    if (e.button === 0 && mouse.inside) mouse.down = true;
  };
  const onMouseUp = (e: MouseEvent) => {
    if (e.button === 0) mouse.down = false;
  };

  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
  canvas.addEventListener("mousemove", onMouseMove);
  window.addEventListener("mousedown", onMouseDown);
  window.addEventListener("mouseup", onMouseUp);

  /** Convert a canvas-relative CSS-pixel point to world coordinates (ortho camera). */
  function screenToWorld(sx: number, sy: number): { x: number; y: number } {
    const rect = canvas.getBoundingClientRect();
    const zoom = cameraZoom.value;
    // Camera is centered on `cameraPosition`; the visible world width is rect.width/zoom.
    // The world's Y axis points down (screen-style), same as screen Y — no Y flip.
    const x = cameraPosition.x + (sx - rect.width / 2) / zoom;
    const y = cameraPosition.y + (sy - rect.height / 2) / zoom;
    return { x, y };
  }

  /** Zero the controllers of the currently held tank (e.g. on disable / switch). */
  function release(targetEid: number) {
    const world = GameDI.world;
    if (!world || !targetEid) return;
    const { Tank, VehicleController, TurretController } = getGameComponents(world);
    if (hasComponent(world, targetEid, VehicleController)) {
      VehicleController.setMove$(targetEid, 0);
      VehicleController.setRotate$(targetEid, 0);
    }
    const turretEid = Tank.turretEId.get(targetEid);
    if (turretEid && hasComponent(world, turretEid, TurretController)) {
      TurretController.setRotation$(turretEid, 0);
      TurretController.setShooting$(turretEid, 0);
    }
  }

  return {
    setEid(next: number) {
      if (next === eid) return;
      release(eid); // stop the tank we are leaving
      eid = next;
    },
    setEnabled(next: boolean) {
      if (next === enabled) return;
      enabled = next;
      if (!enabled) release(eid);
    },
    update(_delta: number) {
      if (!enabled || !eid) return;
      const world = GameDI.world;
      if (!world) return;
      const { Tank, VehicleController, TurretController, RigidBodyState } =
        getGameComponents(world);

      // The tank may have died / the field may have been recreated.
      if (!hasComponent(world, eid, Tank) || !hasComponent(world, eid, VehicleController)) {
        eid = 0;
        return;
      }

      // ── Hull ──────────────────────────────────────────────────────
      const move = (keys.up ? 1 : 0) + (keys.down ? -1 : 0);
      const rotate = (keys.left ? -1 : 0) + (keys.right ? 1 : 0);
      VehicleController.setMove$(eid, move);
      VehicleController.setRotate$(eid, rotate);

      // ── Turret (aim + fire) ───────────────────────────────────────
      const turretEid = Tank.turretEId.get(eid);
      if (!turretEid || !hasComponent(world, turretEid, TurretController)) return;

      const target = screenToWorld(mouse.x, mouse.y);
      const turretX = RigidBodyState.position.get(turretEid, 0);
      const turretY = RigidBodyState.position.get(turretEid, 1);
      const desired = Math.atan2(target.y - turretY, target.x - turretX);
      const err = normalizeAngle(desired - RigidBodyState.rotation[turretEid]);

      if (Math.abs(err) <= AIM_TOLERANCE) {
        TurretController.setRotation$(turretEid, 0);
      } else {
        // Proportional within SLOW_BAND, full speed outside it.
        const dir = Math.abs(err) >= SLOW_BAND ? Math.sign(err) : err / SLOW_BAND;
        TurretController.setRotation$(turretEid, dir);
      }

      // Hold-to-fire: the bullet system rate-limits via reload, so a held
      // button just keeps the shoot flag raised.
      TurretController.setShooting$(turretEid, mouse.down ? 1 : 0);
    },
    dispose() {
      release(eid);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      canvas.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mouseup", onMouseUp);
    },
  };
}
