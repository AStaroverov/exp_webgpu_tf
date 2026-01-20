import { EntityId, innerQuery } from "bitecs";
import { GameDI } from "../../DI/GameDI";
import { Cell } from "../Components/Cell";
import { NestedArray } from "renderer/src/utils";

export function createLifeSystem({ world, cells, rows } = GameDI) {
    const copyGrid = NestedArray.f64(cells, rows);
    
    return () => {
        const eids = innerQuery(world, [Cell]);

        for (let i = 0; i < eids.length; i++) {
            const eid = eids[i];
            const x = Cell.x[eid];
            const y = Cell.y[eid];
            let aliveNeighbors = 0;
            forEachNeighbors(Cell.grid, x, y, (neighborEid: EntityId) => {
                if (neighborEid === 0) return;
                aliveNeighbors++;
            });
            if (aliveNeighbors === 2 || aliveNeighbors === 3) {
                copyGrid.set(x, y, eid);
            } else {
                copyGrid.set(x, y, 0);
            }
        }
    };
}

function forEachNeighbors(grid: NestedArray<Float64ArrayConstructor>, x: number, y: number, callback: (eid: EntityId) => void) {
    for (let i = -1; i <= 1; i++) {
        for (let j = -1; j <= 1; j++) {
            if (i === 0 && j === 0) continue;
            callback(grid.get(x + i, y + j));
        }
    }
}