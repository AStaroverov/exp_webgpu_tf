/**
 * HexGrid — the map grid.
 *
 * Built on `honeycomb-grid`: holds the hex geometry/topology (centers, corners,
 * neighbors, distance) plus a per-cell occupancy/walkability store.
 *
 * Occupancy is stored *by reference*: each cell remembers which entity sits on
 * it (`occupantEid`) and what kind of occupant it is (`occupantKind`). The grid
 * itself is NOT an ECS world — wiring grid cells to ECS entities is a later step.
 */

import { EntityId } from 'bitecs';
import { Grid, rectangle, type HexCoordinates } from 'honeycomb-grid';
import { HexGridConfig, HexTile, POINTY_DIRECTIONS } from './HexConfig.ts';

/**
 * What kind of entity occupies a hex. A cell is blocked the same way regardless
 * of kind; the kind lets consumers tell a movable unit apart from a static
 * obstacle without resolving the entity in its world.
 */
export enum OccupantKind {
    /** A movable entity (a tank). */
    Unit = 0,
    /** A static obstacle (rock / building). */
    Obstacle = 1,
    /** A cell a unit is *driving into* — not yet physically occupied, but no longer free. */
    Reserved = 2,
}

export type HexCell = {
    readonly q: number;
    readonly r: number;
    /** Entity currently occupying this cell, or null. */
    occupantEid: EntityId | null;
    /** What kind of entity occupies this cell, or null when empty. */
    occupantKind: OccupantKind | null;
};

export const cellKey = (q: number, r: number): string => `${q},${r}`;

export class HexGrid {
    /** honeycomb grid — source of truth for geometry & which hexes exist. */
    readonly grid: Grid<HexTile>;

    /** World-space offset added to honeycomb hex centers to place the grid. */
    readonly originX: number;
    readonly originY: number;

    private readonly cells = new Map<string, HexCell>();
    /** Reverse index: `row:col` → the honeycomb hex at that grid position. */
    private readonly hexByRowCol = new Map<string, HexTile>();

    constructor(opts?: {
        originX?: number;
        originY?: number;
        center?: { x: number; y: number };
        cols?: number;
        rows?: number;
    }) {
        this.grid = new Grid(
            HexTile,
            rectangle({
                width: opts?.cols ?? HexGridConfig.cols,
                height: opts?.rows ?? HexGridConfig.rows,
            }),
        );

        if (opts?.center) {
            // Place the grid so its geometric center sits at `center`.
            let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            this.grid.forEach((hex) => {
                if (hex.x < minX) minX = hex.x;
                if (hex.y < minY) minY = hex.y;
                if (hex.x > maxX) maxX = hex.x;
                if (hex.y > maxY) maxY = hex.y;
            });
            this.originX = opts.center.x - (minX + maxX) / 2;
            this.originY = opts.center.y - (minY + maxY) / 2;
        } else {
            this.originX = opts?.originX ?? 0;
            this.originY = opts?.originY ?? 0;
        }

        this.grid.forEach((hex) => {
            this.cells.set(cellKey(hex.q, hex.r), {
                q: hex.q,
                r: hex.r,
                occupantEid: null,
                occupantKind: null,
            });
            this.hexByRowCol.set(cellKey(hex.row, hex.col), hex);
        });
    }

    /** Number of hexes in the grid. */
    get size(): number {
        return this.cells.size;
    }

    has(coord: HexCoordinates): boolean {
        return this.grid.getHex(coord) !== undefined;
    }

    getCell(q: number, r: number): HexCell | undefined {
        return this.cells.get(cellKey(q, r));
    }

    /**
     * Hex at a given grid `(row, col)` position, or undefined if out of bounds.
     * Layout matches the board observation (`row * cols + col`, cell-major), so a
     * flat cell index `idx` decodes as `row = idx / cols`, `col = idx % cols`.
     */
    cellAt(row: number, col: number): HexTile | undefined {
        return this.hexByRowCol.get(cellKey(row, col));
    }

    forEachCell(fn: (cell: HexCell, hex: HexTile) => void): void {
        this.grid.forEach((hex) => {
            const cell = this.cells.get(cellKey(hex.q, hex.r));
            if (cell) fn(cell, hex);
        });
    }

    // --- geometry -----------------------------------------------------------

    /** World-space center of a hex. */
    hexToWorld(q: number, r: number): { x: number; y: number } | undefined {
        const hex = this.grid.getHex({ q, r });
        if (!hex) return undefined;
        return { x: hex.x + this.originX, y: hex.y + this.originY };
    }

    /** Hex containing a world-space point (or undefined if outside the grid). */
    worldToHex(x: number, y: number): HexTile | undefined {
        return this.grid.pointToHex(
            { x: x - this.originX, y: y - this.originY },
            { allowOutside: false },
        );
    }

    /** World-space corner points for rendering a hex outline. */
    cornersOf(coord: HexCoordinates): Array<{ x: number; y: number }> | undefined {
        const hex = this.grid.getHex(coord);
        if (!hex) return undefined;
        return hex.corners.map((p) => ({ x: p.x + this.originX, y: p.y + this.originY }));
    }

    /** World-space bounding box of the whole grid (including hex corners). */
    worldBounds(): { minX: number; minY: number; maxX: number; maxY: number } {
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        this.grid.forEach((hex) => {
            for (const p of hex.corners) {
                const x = p.x + this.originX;
                const y = p.y + this.originY;
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
            }
        });
        return { minX, minY, maxX, maxY };
    }

    /**
     * March a straight ray from a world-space point, visiting each distinct hex
     * it crosses in order. The sample stride is half the hex radius (below the
     * inradius), so no cell is skipped. The walk stops at the grid edge (a
     * straight ray never re-enters), after `maxDistance` world units, or when
     * `visit` returns false. The starting cell is visited with `isFirst = true`
     * so callers can skip it or exempt it from blocking.
     */
    raycast(
        startX: number,
        startY: number,
        dirX: number,
        dirY: number,
        maxDistance: number,
        visit: (cell: HexCell, hex: HexTile, isFirst: boolean) => boolean,
    ): void {
        const stride = HexGridConfig.radius * 0.5;
        let lastQ = NaN;
        let lastR = NaN;
        let first = true;
        for (let t = 0; t <= maxDistance; t += stride) {
            const hex = this.worldToHex(startX + dirX * t, startY + dirY * t);
            if (!hex) return;
            if (hex.q === lastQ && hex.r === lastR) continue;
            lastQ = hex.q;
            lastR = hex.r;
            const cell = this.cells.get(cellKey(hex.q, hex.r))!;
            if (!visit(cell, hex, first)) return;
            first = false;
        }
    }

    /** Hex distance (number of steps) between two cells. */
    distance(a: HexCoordinates, b: HexCoordinates): number {
        return this.grid.distance(a, b);
    }

    neighbors(coord: HexCoordinates): HexTile[] {
        const result: HexTile[] = [];
        for (const dir of POINTY_DIRECTIONS) {
            const n = this.grid.neighborOf(coord, dir, { allowOutside: false });
            if (n) result.push(n);
        }
        return result;
    }

    neighborAt(coord: HexCoordinates, dirIndex: number): HexTile | undefined {
        const dir = POINTY_DIRECTIONS[dirIndex];
        if (dir === undefined) return undefined;
        return this.grid.neighborOf(coord, dir, { allowOutside: false }) ?? undefined;
    }

    // --- occupancy ----------------------------------------------------------

    /**
     * A cell can be entered if it exists and is currently empty. Any occupant —
     * `Unit`, `Obstacle`, or `Reserved` — makes the cell impassable.
     */
    isPassable(q: number, r: number): boolean {
        const cell = this.cells.get(cellKey(q, r));
        return cell != null && cell.occupantEid === null;
    }

    occupy(q: number, r: number, eid: EntityId, kind: OccupantKind): void {
        const cell = this.cells.get(cellKey(q, r));
        if (!cell) return;
        cell.occupantEid = eid;
        cell.occupantKind = kind;
    }

    vacate(q: number, r: number): void {
        const cell = this.cells.get(cellKey(q, r));
        if (!cell) return;
        cell.occupantEid = null;
        cell.occupantKind = null;
    }

    getOccupant(q: number, r: number): { eid: EntityId; kind: OccupantKind } | null {
        const cell = this.cells.get(cellKey(q, r));
        if (!cell || cell.occupantEid === null || cell.occupantKind === null) return null;
        return { eid: cell.occupantEid, kind: cell.occupantKind };
    }
}
