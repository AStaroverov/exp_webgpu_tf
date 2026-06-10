/**
 * Debug control panel (lil-gui):
 *   - Field:    recreate the whole game with a fresh obstacle layout;
 *   - Spawn:    pick a team + optional q,r, then a button per vehicle type
 *               (empty q/r spawns on a random free cell);
 *   - Vehicles: pick a living vehicle; each action kind is its own section with
 *               its settings and an Apply button.
 *
 * Every handler re-reads `GameDI.world` so the panel survives field recreation.
 */

import GUI, { Controller } from "lil-gui";
import { hasComponent, query } from "bitecs";
import { GameDI } from "../../../unknown/src/Game/DI/GameDI.ts";
import { MapDI } from "../../../unknown/src/Game/DI/MapDI.ts";
import { getGameComponents } from "../../../unknown/src/Game/ECS/createGameWorld.ts";
import {
  createTank,
  type TankVehicleType,
} from "../../../unknown/src/Game/ECS/Entities/Tank/createTank.ts";
import {
  destroyTank,
  getTankTeamId,
} from "../../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts";
import { enqueueAction } from "../../../unknown/src/Game/ECS/Actions/ActionSchedule.ts";
import { ActionKind, TargetKind } from "../../../unknown/src/Game/ECS/Actions/ActionTypes.ts";
import { VehicleType, TEAM_BASE_COLORS } from "../../../unknown/src/Game/Config/index.ts";
import { setCameraZoom } from "../../../renderer/src/ECS/Systems/ResizeSystem.ts";
import { DEFAULT_FIELD_SIZE, recreateDebugGame } from "../createDebugGame.ts";
import { spawnObstacles } from "../../../unknown/src/Game/ECS/Entities/Obstacle/spawnObstacles.ts";
import { createLightingGUI } from "../../../unknown/src/ui/createLightingGUI.ts";
import type { ManualControl } from "../input/createManualControl.ts";

// Team identity colors — sourced from the shared vehicle palette so the debug
// swatches and spawned tanks match.
const TEAM_COLORS: Record<number, Float32Array> = TEAM_BASE_COLORS;

const HOLD_DURATION_MS = 1000;
const AIM_TOLERANCE = 0.05;

export function createDebugGUI(canvas: HTMLCanvasElement, manualControl?: ManualControl) {
  let nextPlayerId = 1;
  const gui = new GUI({ title: "Debug" });
  const camera = { zoom: 0.5 };
  function applyZoom() {
    const grid = MapDI.grid;
    if (!grid) return;
    const bounds = grid.worldBounds();
    const margin = 1.2; // keep in sync with createGame's auto-fit
    const fit = Math.min(
      GameDI.width / ((bounds.maxX - bounds.minX) * margin),
      GameDI.height / ((bounds.maxY - bounds.minY) * margin),
    );
    setCameraZoom(fit * camera.zoom);
  }
  applyZoom();

  // ── Lighting ──────────────────────────────────────────────────────────
  // Own lil-gui window, pinned left so it clears the Debug panel (top-right).
  const lighting = createLightingGUI({ container: document.body, side: "left" });

  // ── Field ─────────────────────────────────────────────────────────────
  // Field starts empty at the chosen size; obstacles are opt-in (button).
  const field = { cols: DEFAULT_FIELD_SIZE.cols, rows: DEFAULT_FIELD_SIZE.rows };
  const fieldFolder = gui.addFolder("Field");
  fieldFolder.add(field, "cols", 2, 30, 1).name("cols");
  fieldFolder.add(field, "rows", 2, 30, 1).name("rows");
  fieldFolder
    .add(
      {
        recreate: async () => {
          await recreateDebugGame(canvas, { cols: field.cols, rows: field.rows });
          applyZoom(); // createGame resets the zoom to the auto-fit value
          lighting.sync(); // re-apply panel values to the freshly created lighting system
        },
      },
      "recreate",
    )
    .name("Recreate field");
  fieldFolder.add({ go: () => spawnObstacles() }, "go").name("Spawn obstacles");

  // ── Camera ────────────────────────────────────────────────────────────
  // Multiplier on top of the auto-fit zoom createGame computes (>1 closer).
  gui.add(camera, "zoom", 0.2, 3, 0.05).name("Camera zoom").onChange(applyZoom);

  // ── Spawn ─────────────────────────────────────────────────────────────
  const spawn = {
    team: 1,
    q: "", // empty → random free cell
    r: "",
  };
  const spawnFolder = gui.addFolder("Spawn");
  addTeamSwitcher(spawnFolder, spawn);
  spawnFolder.add(spawn, "q");
  spawnFolder.add(spawn, "r");

  // One button per vehicle type — spawns it for the chosen team at q,r
  // (or a random free cell when q/r are left empty).
  const SPAWN_TYPES: Array<[string, TankVehicleType]> = [
    ["Light Tank", VehicleType.LightTank],
    ["Medium Tank", VehicleType.MediumTank],
    ["Heavy Tank", VehicleType.HeavyTank],
    ["Rocket Tank", VehicleType.RocketTank],
    ["Frost Tank", VehicleType.FrostTank],
    ["Flame Tank", VehicleType.FlameTank],
  ];
  for (const [label, type] of SPAWN_TYPES) {
    spawnFolder
      .add({ go: () => spawnVehicle(type, spawn.team, spawn.q, spawn.r, nextPlayerId++) }, "go")
      .name(label);
  }

  // ── Duel ──────────────────────────────────────────────────────────────
  // Spawn two tanks (team 1 vs team 2) facing each other, `distance` hexes
  // apart along the E-W axis, centered on the grid.
  const duel = { type: VehicleType.MediumTank as TankVehicleType, distance: 3 };
  const duelFolder = gui.addFolder("Duel");
  duelFolder.add(duel, "type", Object.fromEntries(SPAWN_TYPES)).name("vehicle");
  duelFolder.add(duel, "distance", 1, 20, 1).name("distance (hexes)");
  duelFolder
    .add(
      {
        go: () => {
          nextPlayerId = spawnDuel(duel.type, duel.distance, nextPlayerId);
        },
      },
      "go",
    )
    .name("Spawn duel");

  // ── Vehicles ──────────────────────────────────────────────────────────
  const veh = { eid: 0 };
  const vehFolder = gui.addFolder("Vehicles");
  // The eid dropdown lives in its own folder: lil-gui's `options()` replaces
  // the controller by appending a new one, which would otherwise reorder rows.
  const pickFolder = vehFolder.addFolder("Selected");
  let eidCtrl: Controller = pickFolder
    .add(veh, "eid", {})
    .onChange((v: number) => manualControl?.setEid(v));
  manualControl?.setEid(veh.eid);
  vehFolder
    .add(
      {
        go: () => {
          if (veh.eid) destroyTank(veh.eid);
        },
      },
      "go",
    )
    .name("Remove vehicle");

  // ── Manual control (keyboard + mouse) ─────────────────────────────────
  // Drives the *selected* vehicle directly (bypassing the action queue):
  // arrows = hull move/turn, mouse = turret aim, left mouse button = fire.
  // Don't enqueue actions on a tank while it is under manual control.
  if (manualControl) {
    const manual = { enabled: false };
    vehFolder
      .add(manual, "enabled")
      .name("⌨/🖱 control selected")
      .onChange((v: boolean) => manualControl.setEnabled(v));
  }

  // One section per action kind, each with its own settings + an Apply button.
  // MoveStep/Aim/Fire are directional — `dir` indexes POINTY_DIRECTIONS (the
  // target hex is the neighbour in that direction).
  const move = { dir: 0, speed: 1 };
  const moveFolder = vehFolder.addFolder("MoveStep");
  addDirectionSwitcher(moveFolder, move);
  moveFolder.add(move, "speed", 0.1, 3, 0.1).name("speed");
  moveFolder
    .add(
      {
        go: () => {
          if (veh.eid) doMoveStep(veh.eid, move.dir, move.speed);
        },
      },
      "go",
    )
    .name("Apply MoveStep");

  const aim = { dir: 0, tolerance: AIM_TOLERANCE };
  const aimFolder = vehFolder.addFolder("Aim");
  addDirectionSwitcher(aimFolder, aim);
  aimFolder.add(aim, "tolerance", 0.01, 0.5, 0.01).name("tolerance");
  aimFolder
    .add(
      {
        go: () => {
          if (veh.eid) doAim(veh.eid, aim.dir, aim.tolerance);
        },
      },
      "go",
    )
    .name("Apply Aim");

  const fire = { dir: 0 };
  const fireFolder = vehFolder.addFolder("Fire");
  addDirectionSwitcher(fireFolder, fire);
  fireFolder
    .add(
      {
        go: () => {
          if (veh.eid) doFire(veh.eid, fire.dir);
        },
      },
      "go",
    )
    .name("Apply Fire");

  const hold = { duration: HOLD_DURATION_MS };
  const holdFolder = vehFolder.addFolder("Hold");
  holdFolder.add(hold, "duration", 100, 5000, 100).name("duration (ms)");
  holdFolder
    .add(
      {
        go: () => {
          if (veh.eid) doHold(veh.eid, hold.duration);
        },
      },
      "go",
    )
    .name("Apply Hold");

  // Keep the vehicle dropdown in sync with the world (spawns/deaths/recreate).
  let lastKeys = "";
  setInterval(() => {
    const options = listVehicles();
    const keys = Object.keys(options).join("|");
    if (keys === lastKeys) return;
    lastKeys = keys;

    if (!Object.values(options).includes(veh.eid)) {
      veh.eid = Object.values(options)[0] ?? 0;
      manualControl?.setEid(veh.eid);
    }
    eidCtrl = eidCtrl.options(options).onChange((v: number) => manualControl?.setEid(v));
  }, 300);

  // Collapse every section by default (Debug + Lighting) — open on demand.
  gui.foldersRecursive().forEach((f) => f.close());
  lighting.gui.foldersRecursive().forEach((f) => f.close());

  return gui;
}

/** `label → eid` of living tanks (the vehicles that have an actions queue). */
function listVehicles(): Record<string, number> {
  const world = GameDI.world;
  if (!world) return {};
  const { Tank, Vehicle, Children } = getGameComponents(world);
  const options: Record<string, number> = {};
  for (const eid of query(world, [Tank, Vehicle, Children])) {
    options[`#${eid} ${VehicleType[Vehicle.type[eid]]} team${getTankTeamId(eid)}`] = eid;
  }
  return options;
}

function spawnVehicle(
  type: TankVehicleType,
  team: number,
  qStr: string,
  rStr: string,
  playerId: number,
) {
  const grid = MapDI.grid;
  if (!grid) return;
  const cell = resolveSpawnCell(qStr, rStr);
  if (!cell) return;
  const pos = grid.hexToWorld(cell.q, cell.r);
  if (!pos) return;

  createTank({
    type,
    playerId,
    teamId: team,
    x: pos.x,
    y: pos.y,
    rotation: Math.random() * Math.PI * 2,
    color: new Float32Array(TEAM_COLORS[team] ?? TEAM_COLORS[1]),
  });
}

/**
 * Spawn two tanks (team 1 vs team 2) `distance` hexes apart on the E-W axis,
 * centered on the grid, each rotated to face the other. Returns the next free
 * player id. DIRECTIONS index: E = 0, W = 3.
 */
function spawnDuel(type: TankVehicleType, distance: number, nextPlayerId: number): number {
  const grid = MapDI.grid;
  if (!grid) return nextPlayerId;

  const bounds = grid.worldBounds();
  const center = grid.worldToHex((bounds.minX + bounds.maxX) / 2, (bounds.minY + bounds.maxY) / 2);
  if (!center) {
    console.warn("[debug] no center cell to spawn the duel on");
    return nextPlayerId;
  }

  // Split the gap around the center: left walks W, right walks E.
  const left = walkHexes(center, 3, Math.floor(distance / 2));
  const right = walkHexes(center, 0, Math.ceil(distance / 2));
  if (!left || !right) {
    console.warn(`[debug] duel distance ${distance} runs off the grid`);
    return nextPlayerId;
  }

  const lPos = grid.hexToWorld(left.q, left.r);
  const rPos = grid.hexToWorld(right.q, right.r);
  if (!lPos || !rPos) return nextPlayerId;

  createTank({
    type,
    playerId: nextPlayerId++,
    teamId: 1,
    x: lPos.x,
    y: lPos.y,
    rotation: Math.atan2(rPos.y - lPos.y, rPos.x - lPos.x),
    color: new Float32Array(TEAM_COLORS[1]),
  });
  createTank({
    type,
    playerId: nextPlayerId++,
    teamId: 2,
    x: rPos.x,
    y: rPos.y,
    rotation: Math.atan2(lPos.y - rPos.y, lPos.x - rPos.x),
    color: new Float32Array(TEAM_COLORS[2]),
  });
  return nextPlayerId;
}

/** Walk `steps` hexes from `start` in direction `dir`, or null if it leaves the grid. */
function walkHexes(
  start: { q: number; r: number },
  dir: number,
  steps: number,
): { q: number; r: number } | null {
  const grid = MapDI.grid;
  if (!grid) return null;
  let cur: { q: number; r: number } = start;
  for (let i = 0; i < steps; i++) {
    const next = grid.neighborAt(cur, dir);
    if (!next) return null;
    cur = { q: next.q, r: next.r };
  }
  return cur;
}

/** Resolve the spawn cell from the q/r inputs: empty → random free cell. */
function resolveSpawnCell(qStr: string, rStr: string): { q: number; r: number } | null {
  const qt = qStr.trim();
  const rt = rStr.trim();
  if (qt === "" || rt === "") {
    const cell = pickRandomFreeCell();
    if (!cell) console.warn("[debug] no free cell to spawn on");
    return cell;
  }
  const q = Math.round(Number(qt));
  const r = Math.round(Number(rt));
  if (!Number.isFinite(q) || !Number.isFinite(r)) {
    console.warn(`[debug] invalid q/r: "${qStr}", "${rStr}"`);
    return null;
  }
  if (!MapDI.grid?.hexToWorld(q, r)) {
    console.warn(`[debug] hex (${q}, ${r}) is off the grid`);
    return null;
  }
  return { q, r };
}

function pickRandomFreeCell(): { q: number; r: number } | null {
  const grid = MapDI.grid;
  if (!grid) return null;
  const free: Array<{ q: number; r: number }> = [];
  grid.forEachCell((cell) => {
    if (grid.isPassable(cell.q, cell.r)) free.push({ q: cell.q, r: cell.r });
  });
  return free.length > 0 ? free[Math.floor(Math.random() * free.length)] : null;
}

/** A row of colored team buttons (the active one highlighted) — a compact team switcher. */
function addTeamSwitcher(folder: GUI, target: { team: number }) {
  const row = document.createElement("div");
  Object.assign(row.style, {
    display: "flex",
    gap: "4px",
    padding: "6px 8px",
    alignItems: "center",
  });

  const label = document.createElement("span");
  label.textContent = "team";
  Object.assign(label.style, { flex: "1", fontSize: "11px" });
  row.appendChild(label);

  const teams = Object.keys(TEAM_COLORS).map(Number);
  const buttons: HTMLButtonElement[] = [];
  const refresh = () =>
    buttons.forEach((b, i) => {
      const active = teams[i] === target.team;
      b.style.outline = active ? "2px solid #fff" : "none";
      b.style.opacity = active ? "1" : "0.4";
    });

  for (const t of teams) {
    const [r, g, b] = TEAM_COLORS[t];
    const btn = document.createElement("button");
    btn.textContent = String(t);
    Object.assign(btn.style, {
      width: "22px",
      height: "22px",
      cursor: "pointer",
      border: "none",
      borderRadius: "3px",
      color: "#000",
      fontWeight: "bold",
      background: `rgb(${(r * 255) | 0}, ${(g * 255) | 0}, ${(b * 255) | 0})`,
    });
    btn.addEventListener("click", () => {
      target.team = t;
      refresh();
    });
    buttons.push(btn);
    row.appendChild(btn);
  }
  refresh();
  folder.$children.appendChild(row);
}

/** Direction labels, indexed to match POINTY_DIRECTIONS (HexGrid.neighborAt). */
const DIRECTIONS = ["E", "NE", "NW", "W", "SW", "SE"] as const;

/** Current hex of a vehicle, or null if it is off the grid. */
function vehicleHex(eid: number): { q: number; r: number } | null {
  const world = GameDI.world;
  const grid = MapDI.grid;
  if (!world || !grid) return null;
  const { RigidBodyState } = getGameComponents(world);
  const px = RigidBodyState.position.get(eid, 0);
  const py = RigidBodyState.position.get(eid, 1);
  const here = grid.worldToHex(px, py);
  if (!here) {
    console.warn(`[debug] vehicle #${eid} is off the grid`);
    return null;
  }
  return { q: here.q, r: here.r };
}

/** Neighbour hex of the vehicle in the chosen direction, or null (off grid). */
function targetHexInDirection(eid: number, dir: number): { q: number; r: number } | null {
  const grid = MapDI.grid;
  const here = vehicleHex(eid);
  if (!grid || !here) return null;
  const dest = grid.neighborAt(here, dir);
  if (!dest) {
    console.warn(`[debug] no hex ${DIRECTIONS[dir]} of vehicle #${eid} (edge of grid)`);
    return null;
  }
  return { q: dest.q, r: dest.r };
}

function doMoveStep(eid: number, dir: number, speed: number) {
  const grid = MapDI.grid;
  const dest = targetHexInDirection(eid, dir);
  if (!grid || !dest) return;
  if (!grid.isPassable(dest.q, dest.r)) {
    console.warn(`[debug] hex ${DIRECTIONS[dir]} of vehicle #${eid} is blocked`);
    return;
  }
  enqueueAction(eid, {
    kind: ActionKind.MoveStep,
    target: { kind: TargetKind.Hex, q: dest.q, r: dest.r },
    params: { speed },
  });
}

function doAim(eid: number, dir: number, tolerance: number) {
  const dest = targetHexInDirection(eid, dir);
  if (!dest) return;
  enqueueAction(eid, {
    kind: ActionKind.Aim,
    target: { kind: TargetKind.Hex, q: dest.q, r: dest.r },
    params: { tolerance },
  });
}

function doFire(eid: number, dir: number) {
  const world = GameDI.world;
  const dest = targetHexInDirection(eid, dir);
  if (!world || !dest) return;
  const { Tank, Firearms } = getGameComponents(world);
  // Fire: unarmed turrets have no Firearms — their Fire action would hang
  // waiting for a reload that never starts.
  if (!hasComponent(world, Tank.turretEId[eid], Firearms)) {
    console.warn(`[debug] vehicle #${eid} is unarmed, Fire skipped`);
    return;
  }
  enqueueAction(eid, {
    kind: ActionKind.Fire,
    target: { kind: TargetKind.Hex, q: dest.q, r: dest.r },
  });
}

function doHold(eid: number, duration: number) {
  enqueueAction(eid, { kind: ActionKind.Hold, params: { duration } });
}

/** A row of the 6 pointy-hex directions (the active one highlighted). */
function addDirectionSwitcher(folder: GUI, target: { dir: number }) {
  const wrap = document.createElement("div");
  Object.assign(wrap.style, { padding: "6px 8px" });

  const label = document.createElement("div");
  label.textContent = "direction";
  Object.assign(label.style, { fontSize: "11px", marginBottom: "4px" });
  wrap.appendChild(label);

  const row = document.createElement("div");
  Object.assign(row.style, { display: "flex", gap: "4px", flexWrap: "wrap" });

  const buttons: HTMLButtonElement[] = [];
  const refresh = () =>
    buttons.forEach((b, i) => {
      const active = i === target.dir;
      b.style.background = active ? "#5b8def" : "#3a3a3a";
      b.style.outline = active ? "2px solid #fff" : "none";
    });

  DIRECTIONS.forEach((name, i) => {
    const btn = document.createElement("button");
    btn.textContent = name;
    Object.assign(btn.style, {
      minWidth: "30px",
      height: "22px",
      cursor: "pointer",
      border: "none",
      borderRadius: "3px",
      color: "#fff",
      fontSize: "11px",
    });
    btn.addEventListener("click", () => {
      target.dir = i;
      refresh();
    });
    buttons.push(btn);
    row.appendChild(btn);
  });
  refresh();
  wrap.appendChild(row);
  folder.$children.appendChild(wrap);
}
