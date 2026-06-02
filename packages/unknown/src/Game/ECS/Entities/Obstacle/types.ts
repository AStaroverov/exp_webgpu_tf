/** Obstacle planning types — shared by the spawn system and the factories. */

/**
 * A planned obstacle: the hex cells (footprint) it occupies plus the anchor it
 * grew from. Produced by prebuild, consumed by the factory at commit. Rocks
 * occupy a single cell; the array leaves room for multi-cell obstacles later.
 */
export type ObstaclePlan = {
    anchor: { q: number; r: number };
    cells: Array<{ q: number; r: number }>;
};
