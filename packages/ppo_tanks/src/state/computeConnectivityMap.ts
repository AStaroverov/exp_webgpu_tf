import { GRID_SIZE } from '../models/dims.ts';

export const CONNECTIVITY_BFS_DEPTH = 5;
const OBSTACLE_THRESHOLD = 0.5;

// 8-directional neighbors: [dRow, dCol]
const DR = [-1, -1, -1, 0, 0, 1, 1, 1];
const DC = [-1, 0, 1, -1, 1, -1, 0, 1];

function isBoundary(row: number, col: number): boolean {
    return row === 0 || row === GRID_SIZE - 1 || col === 0 || col === GRID_SIZE - 1;
}

/**
 * Connectivity heat map for the obstacle grid.
 * For each free cell, bounded BFS (depth ≤ CONNECTIVITY_BFS_DEPTH)
 * counts how many cells are reachable — open fields score high,
 * corridors score medium, dead-ends score low.
 * Result is normalized to [0, 1].
 * Computed once per episode (obstacles are static).
 */
export function computeConnectivityMap(
    obstacleGrid: Float32Array,
): Float32Array {
    const total = GRID_SIZE * GRID_SIZE;
    const connectivity = new Float32Array(total);

    // Reusable BFS buffers
    const visited = new Uint8Array(total);
    const queue = new Int32Array(total);
    const depth = new Uint8Array(total);

    let maxConn = 0;

    for (let start = 0; start < total; start++) {
        const startRow = (start / GRID_SIZE) | 0;
        const startCol = start % GRID_SIZE;
        if (obstacleGrid[start] > OBSTACLE_THRESHOLD || isBoundary(startRow, startCol)) continue;

        visited.fill(0);
        visited[start] = 1;
        depth[start] = 0;
        queue[0] = start;
        let head = 0;
        let tail = 1;
        let count = 0;

        while (head < tail) {
            const idx = queue[head++];
            const d = depth[idx];
            count++;

            if (d >= CONNECTIVITY_BFS_DEPTH) continue;

            const row = (idx / GRID_SIZE) | 0;
            const col = idx % GRID_SIZE;

            for (let n = 0; n < 8; n++) {
                const nr = row + DR[n];
                const nc = col + DC[n];
                if ((nr | nc) < 0 || nr >= GRID_SIZE || nc >= GRID_SIZE) continue;

                const nIdx = nr * GRID_SIZE + nc;
                if (visited[nIdx]) continue;
                if (obstacleGrid[nIdx] > OBSTACLE_THRESHOLD || isBoundary(nr, nc)) continue;

                visited[nIdx] = 1;
                depth[nIdx] = d + 1;
                queue[tail++] = nIdx;
            }
        }

        connectivity[start] = count;
        if (count > maxConn) maxConn = count;
    }

    // Normalize to [0, 1]
    if (maxConn > 0) {
        for (let i = 0; i < total; i++) {
            connectivity[i] /= maxConn;
        }
    }

    return connectivity;
}
