/**
 * Debug control panel (lil-gui):
 *   - Field:    recreate the whole game with a fresh obstacle layout;
 *   - Spawn:    add a vehicle of the chosen type/team onto a random free cell;
 *   - Vehicles: pick a living vehicle, enqueue an action for it, or remove it.
 *
 * Every handler re-reads `GameDI.world` so the panel survives field recreation.
 */

import GUI, { Controller } from 'lil-gui';
import { hasComponent, query } from 'bitecs';
import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { MapDI } from '../../../unknown/src/Game/DI/MapDI.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';
import { createTank, type TankVehicleType } from '../../../unknown/src/Game/ECS/Entities/Tank/createTank.ts';
import { destroyTank, getTankTeamId } from '../../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { enqueueAction } from '../../../unknown/src/Game/ECS/Actions/ActionSchedule.ts';
import { ActionKind, TargetKind } from '../../../unknown/src/Game/ECS/Actions/ActionTypes.ts';
import { VehicleType } from '../../../unknown/src/Game/Config/index.ts';
import { setCameraZoom } from '../../../renderer/src/ECS/Systems/ResizeSystem.ts';
import { recreateDebugGame } from '../createDebugGame.ts';

const TEAM_COLORS: Record<number, [number, number, number, number]> = {
    1: [1.0, 0.4, 0.4, 1],
    2: [0.4, 0.7, 1.0, 1],
    3: [0.6, 1.0, 0.5, 1],
    4: [1.0, 0.9, 0.4, 1],
};

const HOLD_DURATION_MS = 1000;
const AIM_TOLERANCE = 0.05;

export function createDebugGUI(canvas: HTMLCanvasElement) {
    let nextPlayerId = 1;
    const gui = new GUI({ title: 'Debug' });
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

    // ── Field ─────────────────────────────────────────────────────────────
    gui.add({
        recreate: async () => {
            await recreateDebugGame(canvas);
            applyZoom(); // createGame resets the zoom to the auto-fit value
        },
    }, 'recreate').name('Recreate field');

    // ── Camera ────────────────────────────────────────────────────────────
    // Multiplier on top of the auto-fit zoom createGame computes (>1 closer).
    gui.add(camera, 'zoom', 0.2, 3, 0.05).name('Camera zoom').onChange(applyZoom);
    
    // ── Spawn ─────────────────────────────────────────────────────────────
    const spawn = {
        type: VehicleType.LightTank as TankVehicleType,
        team: 1,
        spawn() {
            const cell = pickRandomFreeCell();
            if (!cell) {
                console.warn('[debug] no free cell to spawn on');
                return;
            }
            const pos = MapDI.grid.hexToWorld(cell.q, cell.r);
            if (!pos) return;

            createTank({
                type: spawn.type,
                playerId: nextPlayerId++,
                teamId: spawn.team,
                x: pos.x,
                y: pos.y,
                rotation: Math.random() * Math.PI * 2,
                color: new Float32Array(TEAM_COLORS[spawn.team] ?? TEAM_COLORS[1]),
            });
        },
    };
    const spawnFolder = gui.addFolder('Spawn');
    spawnFolder.add(spawn, 'type', {
        LightTank: VehicleType.LightTank,
        MediumTank: VehicleType.MediumTank,
        HeavyTank: VehicleType.HeavyTank,
    });
    spawnFolder.add(spawn, 'team', Object.keys(TEAM_COLORS).map(Number));
    spawnFolder.add(spawn, 'spawn').name('Spawn at random cell');

    // ── Vehicles ──────────────────────────────────────────────────────────
    const veh = {
        eid: 0,
        action: ActionKind.MoveStep,
        apply() {
            if (veh.eid) applyAction(veh.eid, veh.action);
        },
        remove() {
            if (veh.eid) destroyTank(veh.eid);
        },
    };
    const vehFolder = gui.addFolder('Vehicles');
    // The eid dropdown lives in its own folder: lil-gui's `options()` replaces
    // the controller by appending a new one, which would otherwise reorder rows.
    const pickFolder = vehFolder.addFolder('Selected');
    let eidCtrl: Controller = pickFolder.add(veh, 'eid', {});
    vehFolder.add(veh, 'action', {
        MoveStep: ActionKind.MoveStep,
        Aim: ActionKind.Aim,
        Fire: ActionKind.Fire,
        Hold: ActionKind.Hold,
    });
    vehFolder.add(veh, 'apply').name('Apply action');
    vehFolder.add(veh, 'remove').name('Remove vehicle');

    // Keep the vehicle dropdown in sync with the world (spawns/deaths/recreate).
    let lastKeys = '';
    setInterval(() => {
        const options = listVehicles();
        const keys = Object.keys(options).join('|');
        if (keys === lastKeys) return;
        lastKeys = keys;

        if (!Object.values(options).includes(veh.eid)) {
            veh.eid = Object.values(options)[0] ?? 0;
        }
        eidCtrl = eidCtrl.options(options);
    }, 300);

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

function pickRandomFreeCell(): { q: number; r: number } | null {
    const grid = MapDI.grid;
    if (!grid) return null;
    const free: Array<{ q: number; r: number }> = [];
    grid.forEachCell((cell) => {
        if (grid.isPassable(cell.q, cell.r)) free.push({ q: cell.q, r: cell.r });
    });
    return free.length > 0 ? free[Math.floor(Math.random() * free.length)] : null;
}

function applyAction(eid: number, kind: ActionKind) {
    const world = GameDI.world;
    const grid = MapDI.grid;
    if (!world || !grid) return;
    const { Tank, Firearms, RigidBodyState } = getGameComponents(world);

    if (kind === ActionKind.Hold) {
        enqueueAction(eid, { kind, params: { duration: HOLD_DURATION_MS } });
        return;
    }

    const px = RigidBodyState.position.get(eid, 0);
    const py = RigidBodyState.position.get(eid, 1);
    const here = grid.worldToHex(px, py);
    if (!here) {
        console.warn(`[debug] vehicle #${eid} is off the grid`);
        return;
    }

    if (kind === ActionKind.MoveStep) {
        const passable = grid
            .neighbors({ q: here.q, r: here.r })
            .filter((n) => grid.isPassable(n.q, n.r));
        if (passable.length === 0) {
            console.warn(`[debug] vehicle #${eid} is hemmed in, nowhere to move`);
            return;
        }
        const dest = passable[Math.floor(Math.random() * passable.length)];
        enqueueAction(eid, {
            kind,
            target: { kind: TargetKind.Hex, q: dest.q, r: dest.r },
            params: { speed: 1 },
        });
        return;
    }

    // Aim / Fire — point at the nearest other vehicle, or a random neighbour.
    const target = nearestOtherVehicleHex(eid, px, py)
        ?? pickRandomNeighbor(here.q, here.r);
    if (!target) {
        console.warn(`[debug] vehicle #${eid} has nothing to aim at`);
        return;
    }

    if (kind === ActionKind.Aim) {
        enqueueAction(eid, {
            kind,
            target: { kind: TargetKind.Hex, q: target.q, r: target.r },
            params: { tolerance: AIM_TOLERANCE },
        });
        return;
    }

    // Fire: unarmed turrets have no Firearms — their Fire action would hang
    // waiting for a reload that never starts.
    if (!hasComponent(world, Tank.turretEId[eid], Firearms)) {
        console.warn(`[debug] vehicle #${eid} is unarmed, Fire skipped`);
        return;
    }
    enqueueAction(eid, {
        kind: ActionKind.Fire,
        target: { kind: TargetKind.Hex, q: target.q, r: target.r },
    });
}

function pickRandomNeighbor(q: number, r: number): { q: number; r: number } | null {
    const neighbors = MapDI.grid.neighbors({ q, r });
    return neighbors.length > 0
        ? neighbors[Math.floor(Math.random() * neighbors.length)]
        : null;
}

/** Hex of the closest other living vehicle, or null if there is none. */
function nearestOtherVehicleHex(selfEid: number, px: number, py: number): { q: number; r: number } | null {
    const world = GameDI.world;
    const { Tank, Vehicle, Children, RigidBodyState } = getGameComponents(world);
    let bestHex: { q: number; r: number } | null = null;
    let bestDist = Infinity;
    for (const other of query(world, [Tank, Vehicle, Children])) {
        if (other === selfEid) continue;
        const ox = RigidBodyState.position.get(other, 0);
        const oy = RigidBodyState.position.get(other, 1);
        const d = (ox - px) * (ox - px) + (oy - py) * (oy - py);
        if (d < bestDist) {
            const hex = MapDI.grid.worldToHex(ox, oy);
            if (hex) {
                bestDist = d;
                bestHex = { q: hex.q, r: hex.r };
            }
        }
    }
    return bestHex;
}
