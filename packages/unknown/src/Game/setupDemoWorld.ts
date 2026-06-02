/**
 * setupDemoWorld — build-specific world content for the dev/demo game.
 *
 * Extracted out of `createGame` so the base game wires only systems. The demo
 * build adds: static obstacles, four demo tanks on random cells, and the scripted
 * stand-in decision driver (the placeholder the ML policy replaces). Training
 * builds (`ppo_unknown`) skip this entirely and spawn their own teams + policy
 * driver via `createUnknownScenario`.
 *
 * Must be called AFTER `createGame()` (it relies on the live `GameDI`/`MapDI`).
 */

import { GameDI } from './DI/GameDI.ts';
import { MapDI } from './DI/MapDI.ts';
import { PluginDI } from './DI/PluginDI.ts';
import { getGameComponents } from './ECS/createGameWorld.ts';
import { SystemGroup } from './ECS/Plugins/systems.ts';
import { createTank } from './ECS/Entities/Tank/createTank.ts';
import { VehicleType } from './Config/index.ts';
import { spawnObstacles } from './ECS/Entities/Obstacle/spawnObstacles.ts';
import { createStandInDriverSystem } from './ECS/Plugins/createStandInDriverSystem.ts';
import { createShapeCountDiagnosticSystem } from './ECS/Plugins/createShapeCountDiagnosticSystem.ts';

export function setupDemoWorld({ world } = GameDI) {
    const { VehicleController } = getGameComponents(world);

    // Stand-in decision driver — placeholder for the future ML policy. Runs in
    // SystemGroup.Before so decisions land before the gameplay/spawn systems act
    // on them this tick. Same seam the ML driver uses.
    PluginDI.addSystem(SystemGroup.Before, createStandInDriverSystem());

    // TEMP diagnostic (DELETE once the 10k shape-buffer overflow is diagnosed).
    PluginDI.addSystem(SystemGroup.After, createShapeCountDiagnosticSystem());

    spawnObstacles();
    spawnDemoTanks();

    function spawnDemoTanks() {
        const grid = MapDI.grid;
        const palette: Array<[number, number, number, number]> = [
            [1.0, 0.4, 0.4, 1],
            [0.4, 0.7, 1.0, 1],
            [0.6, 1.0, 0.5, 1],
            [1.0, 0.9, 0.4, 1],
        ];

        // Pick distinct random cells to place the tanks on.
        const allCells: Array<{ q: number; r: number }> = [];
        grid.forEachCell((cell) => allCells.push({ q: cell.q, r: cell.r }));
        for (let i = allCells.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [allCells[i], allCells[j]] = [allCells[j], allCells[i]];
        }
        const slots = allCells.slice(0, palette.length);

        for (let i = 0; i < slots.length; i++) {
            const { q, r } = slots[i];
            const pos = grid.hexToWorld({ q, r });
            if (!pos) continue;

            const tankEid = createTank({
                type: VehicleType.MediumTank,
                playerId: i + 1,
                teamId: (i % 2) + 1,
                x: pos.x,
                y: pos.y,
                rotation: Math.random() * Math.PI * 2,
                color: new Float32Array(palette[i % palette.length]),
            });

            VehicleController.setMove$(tankEid, 0);
            VehicleController.setRotate$(tankEid, 0);
        }
    }
}
