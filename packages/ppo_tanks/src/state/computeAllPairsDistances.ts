import { GRID_SIZE, GRID_CELLS } from '../models/dims.ts';

const OBSTACLE_THRESHOLD = 0.35;
const DR = [-1, -1, -1, 0, 0, 1, 1, 1];
const DC = [-1, 0, 1, -1, 1, -1, 0, 1];

/** Sentinel for unreachable cell pairs (larger than any valid BFS distance). */
export const UNREACHABLE = GRID_CELLS;

/**
 * All-pairs shortest path distances on the obstacle grid.
 * BFS with 8-directional movement, unit cost.
 * Returns flat Float32Array of size GRID_CELLS².
 * Access: dist[from * GRID_CELLS + to].
 * Unreachable pairs have distance = UNREACHABLE.
 */
export function computeAllPairsDistances(obstacleGrid: Float32Array): Float32Array {
    const total = GRID_CELLS * GRID_CELLS;
    const dist = new Float32Array(total);
    dist.fill(UNREACHABLE);

    const queue = new Int32Array(GRID_CELLS);
    const cellDist = new Float32Array(GRID_CELLS);

    for (let source = 0; source < GRID_CELLS; source++) {
        if (obstacleGrid[source] > OBSTACLE_THRESHOLD) continue;

        const offset = source * GRID_CELLS;
        cellDist.fill(UNREACHABLE);
        cellDist[source] = 0;
        dist[offset + source] = 0;

        queue[0] = source;
        let head = 0;
        let tail = 1;

        while (head < tail) {
            const idx = queue[head++];
            const d = cellDist[idx];
            const row = (idx / GRID_SIZE) | 0;
            const col = idx % GRID_SIZE;

            for (let n = 0; n < 8; n++) {
                const nr = row + DR[n];
                const nc = col + DC[n];
                if (nr < 0 || nr >= GRID_SIZE || nc < 0 || nc >= GRID_SIZE) continue;

                const nIdx = nr * GRID_SIZE + nc;
                if (cellDist[nIdx] < UNREACHABLE) continue;
                if (obstacleGrid[nIdx] > OBSTACLE_THRESHOLD) continue;

                const nd = d + 1;
                cellDist[nIdx] = nd;
                dist[offset + nIdx] = nd;
                queue[tail++] = nIdx;
            }
        }
    }

    return dist;
}
