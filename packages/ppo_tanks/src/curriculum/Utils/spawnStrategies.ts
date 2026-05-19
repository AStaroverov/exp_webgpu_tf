import { EntityId, query } from 'bitecs';
import { PI } from '../../../../../lib/math.ts';
import { randomRangeFloat, randomRangeInt } from '../../../../../lib/random.ts';
import {
    CellContent,
    getCellWorldPosition,
    getSpawnGrid,
    isCellEmpty,
    setCellContent,
} from '../../../../tanks/src/Arena/State/Game/SpawnGrid.ts';
import { VehicleType } from '../../../../tanks/src/Game/Config/vehicles.ts';
import { createPlayer } from '../../../../tanks/src/Game/ECS/Entities/Player.ts';
import { createTank } from '../../../../tanks/src/Game/ECS/Entities/Tank/createTank.ts';
import { Vehicle } from '../../../../tanks/src/Game/ECS/Components/Vehicle.ts';
import { GameDI } from '../../../../tanks/src/Game/DI/GameDI.ts';
import { addTanksWithGrid } from './addTanksWithGrid.ts';
import { createDiagonalGeometry, createDiagonalTeamTanks, DiagonalTanksResult } from './createDiagonalTanks.ts';

export type SpawnStrategy =
    | { kind: 'grid' }
    | { kind: 'random' }
    | { kind: 'diagonal'; angleNoise?: number };

export type TeamSize = { allies: number; enemies: number };

const tankTypes = [VehicleType.LightTank, VehicleType.MediumTank, VehicleType.HeavyTank] as const;

// Shared with obstacleStrategies so a diagonal wall can align with the diagonal spawn angle.
let lastDiagonalGeometry: DiagonalTanksResult | null = null;
export function getLastDiagonalGeometry(): DiagonalTanksResult | null {
    return lastDiagonalGeometry;
}

export function applySpawn(strategy: SpawnStrategy, teamSize: TeamSize): EntityId[] {
    lastDiagonalGeometry = null;

    switch (strategy.kind) {
        case 'grid': {
            return addTanksWithGrid([[0, teamSize.allies], [1, teamSize.enemies]]);
        }
        case 'random': {
            return spawnRandom(teamSize);
        }
        case 'diagonal': {
            return spawnDiagonal(teamSize, strategy.angleNoise);
        }
    }
}

function spawnRandom(teamSize: TeamSize): EntityId[] {
    const grid = getSpawnGrid();
    const tanks: EntityId[] = [];

    const spawnTank = (teamId: number): EntityId => {
        const totalCells = grid.cols * grid.rows;
        let attempts = 0;
        let col = 0;
        let row = 0;
        do {
            const idx = randomRangeInt(0, totalCells - 1);
            col = idx % grid.cols;
            row = Math.floor(idx / grid.cols);
            attempts++;
        } while (!isCellEmpty(col, row) && attempts < 200);

        const { x, y } = getCellWorldPosition(col, row);
        const tank = createTank({
            type: tankTypes[randomRangeInt(0, tankTypes.length - 1)],
            playerId: createPlayer(teamId),
            teamId,
            x,
            y,
            rotation: PI * randomRangeFloat(0, 2),
            color: teamId === 0
                ? [0, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1]
                : [1, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
        });
        setCellContent(col, row, CellContent.Vehicle, tank);
        return tank;
    };

    for (let i = 0; i < teamSize.allies; i++) tanks.push(spawnTank(0));
    for (let i = 0; i < teamSize.enemies; i++) tanks.push(spawnTank(1));
    return tanks;
}

function spawnDiagonal(teamSize: TeamSize, angleNoise?: number): EntityId[] {
    const fieldSize = GameDI.width;
    const count = Math.max(teamSize.allies, teamSize.enemies);
    const tankOptions = {
        fieldSize,
        edgeMargin: 50,
        maxDeviation: angleNoise ?? PI / 6,
        count,
    };
    const geometry = createDiagonalGeometry(tankOptions);
    lastDiagonalGeometry = geometry;

    // createDiagonalTeamTanks does not return eids, so we snapshot Vehicle query
    // before and after each spawn to recover the newly created tanks in order.
    const before = new Set<EntityId>(query(GameDI.world, [Vehicle]));
    createDiagonalTeamTanks({ ...tankOptions, count: teamSize.allies }, geometry, 0);
    const afterAllies = query(GameDI.world, [Vehicle]);
    const alliesEids = afterAllies.filter((eid) => !before.has(eid));

    const beforeEnemies = new Set<EntityId>(afterAllies);
    createDiagonalTeamTanks({ ...tankOptions, count: teamSize.enemies }, geometry, 1);
    const afterEnemies = query(GameDI.world, [Vehicle]);
    const enemiesEids = afterEnemies.filter((eid) => !beforeEnemies.has(eid));

    return [...alliesEids, ...enemiesEids];
}
