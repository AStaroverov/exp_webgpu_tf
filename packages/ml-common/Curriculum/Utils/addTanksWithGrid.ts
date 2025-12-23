import { PI } from '../../../../lib/math.ts';
import { randomRangeFloat, randomRangeInt } from '../../../../lib/random.ts';
import { createPlayer } from '../../../tanks/src/Game/ECS/Entities/Player.ts';
import { createMediumTank } from '../../../tanks/src/Game/ECS/Entities/Tank/Medium/MediumTank.ts';
import {
    getSpawnGrid,
    getCellWorldPosition,
    setCellContent,
    CellContent,
} from '../../../tanks/src/Arena/State/Game/SpawnGrid.ts';

/**
 * Add tanks at random positions on the grid.
 */
export function addTanksWithGrid(teamIdAndCount: [number, number][]) {
    const tanks = [];
    const grid = getSpawnGrid();
    const totalCells = grid.cols * grid.rows;

    // Generate unique random cell indices
    const usedIndices = new Set<number>();
    const getRandomCell = () => {
        let index: number;
        do {
            index = randomRangeInt(0, totalCells - 1);
        } while (usedIndices.has(index));
        usedIndices.add(index);
        return {
            col: index % grid.cols,
            row: Math.floor(index / grid.cols),
        };
    };

    for (const [teamId, count] of teamIdAndCount) {
        for (let i = 0; i < count; i++) {
            const { col, row } = getRandomCell();
            const { x, y } = getCellWorldPosition(col, row);
            const playerId = createPlayer(teamId);

            const tank = createMediumTank({
                playerId,
                teamId,
                x,
                y,
                rotation: PI * randomRangeFloat(0, 2),
                color: [teamId, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
            });

            setCellContent(col, row, CellContent.Vehicle, tank);
            tanks.push(tank);
        }
    }

    return tanks;
}
