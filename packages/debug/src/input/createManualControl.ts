/**
 * Manual keyboard + mouse control of a single tank, for the debug build.
 *
 *   - Arrow keys drive the hull: Up/Down = forward/back, Left/Right = turn.
 *   - The mouse aims the turret (it tracks the cursor's world position).
 *   - Holding the left mouse button fires (auto-repeats, rate-limited by reload).
 *
 * It bypasses the action queue and writes the same controller components the
 * action executors do (`VehicleController` for the hull, `TurretController` for
 * the turret), so manual input must NOT be mixed with queued actions on the same
 * tank — a queued MoveStep/Aim/Fire runs inside `gameTick` and would overwrite
 * what we set here. `update()` is therefore called once per frame *before*
 * `gameTick`, so the values it writes are consumed by the physics/turret/bullet
 * systems in the very same frame.
 */

import { hasComponent } from "bitecs";
import { GameDI } from "../../../engine/src/Game/DI/GameDI.ts";
import { getGameComponents } from "../../../engine/src/Game/ECS/createGameWorld.ts";
import { normalizeAngle } from "../../../../lib/math.ts";
import { vec3, vec4 } from "gl-matrix";
import {
  cameraPosition,
  invViewProjection,
} from "../../../renderer/src/ECS/Systems/ResizeSystem.ts";

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

  // Pressed state of the four arrow keys.
  const keys = { up: false, down: false, left: false, right: false };
  // Mouse cursor in CSS pixels relative to the canvas, and left-button state.
  const mouse = { x: 0, y: 0, inside: false, down: false };

  const KEY_MAP: Record<string, keyof typeof keys> = {
    ArrowUp: "up",
    ArrowDown: "down",
    ArrowLeft: "left",
    ArrowRight: "right",
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

  // scratch (no per-call alloc)
  const _clip = vec4.create();
  const _far = vec4.create();
  const _rayDir = vec3.create();

  /**
   * Convert a canvas-relative CSS-pixel point to world coordinates by casting a
   * ray through the pixel (perspective camera) and intersecting the gameplay
   * ground plane z=0. Gameplay is still XY-planar, so we solve for the t where
   * the ray hits z=0.
   */
  function screenToWorld(sx: number, sy: number): { x: number; y: number } {
    const rect = canvas.getBoundingClientRect();
    // CSS pixel -> NDC. WebGPU NDC: x in [-1,1] right, y in [-1,1] up (screen Y
    // is down, so flip). Reverse-Z: the far plane (z=0 NDC) reconstructs the ray
    // direction together with the camera eye.
    const ndcX = (sx / rect.width) * 2 - 1;
    const ndcY = 1 - (sy / rect.height) * 2;

    vec4.set(_clip, ndcX, ndcY, 0, 1); // far plane in reverse-Z is NDC z=0
    vec4.transformMat4(_far, _clip, invViewProjection);
    if (_far[3] !== 0) {
      _far[0] /= _far[3];
      _far[1] /= _far[3];
      _far[2] /= _far[3];
    }

    _rayDir[0] = _far[0] - cameraPosition[0];
    _rayDir[1] = _far[1] - cameraPosition[1];
    _rayDir[2] = _far[2] - cameraPosition[2];

    // Intersect with z=0: eye.z + t * dir.z = 0.
    const t = _rayDir[2] !== 0 ? -cameraPosition[2] / _rayDir[2] : 0;
    return {
      x: cameraPosition[0] + t * _rayDir[0],
      y: cameraPosition[1] + t * _rayDir[1],
    };
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
