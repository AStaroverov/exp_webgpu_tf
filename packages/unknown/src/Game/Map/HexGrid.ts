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
 * Worlds whose entities may occupy a hex. Entity id spaces are disjoint between
 * worlds, so an occupant is (world id, entity id).
 */
export enum MapWorldId {
    /** The game ECS world (`GameDI.world`). */
    Game = 0,
}

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
}

export type HexCell = {
    readonly q: number;
    readonly r: number;
    /** Entity currently occupying this cell, or null. */
    occupantEid: EntityId | null;
    /** What kind of entity occupies this cell, or null when empty. */
    occupantKind: OccupantKind | null;
};

const cellKey = (q: number, r: number): string => `${q},${r}`;

export class HexGrid {
    /** honeycomb grid — source of truth for geometry & which hexes exist. */
    readonly grid: Grid<HexTile>;

    /** World-space offset added to honeycomb hex centers to place the grid. */
    readonly originX: number;
    readonly originY: number;

    private readonly cells = new Map<string, HexCell>();

    constructor(opts?: { originX?: number; originY?: number; center?: { x: number; y: number } }) {
        this.grid = new Grid(
            HexTile,
            rectangle({ width: HexGridConfig.cols, height: HexGridConfig.rows }),
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

    forEachCell(fn: (cell: HexCell, hex: HexTile) => void): void {
        this.grid.forEach((hex) => {
            const cell = this.cells.get(cellKey(hex.q, hex.r));
            if (cell) fn(cell, hex);
        });
    }

    // --- geometry -----------------------------------------------------------

    /** World-space center of a hex. */
    hexToWorld(coord: HexCoordinates): { x: number; y: number } | undefined {
        const hex = this.grid.getHex(coord);
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

    /** Hex distance (number of steps) between two cells. */
    distance(a: HexCoordinates, b: HexCoordinates): number {
        return this.grid.distance(a, b);
    }

    /** In-grid neighbors of a cell (1..6 of them). */
    neighbors(coord: HexCoordinates): HexTile[] {
        const result: HexTile[] = [];
        for (const dir of POINTY_DIRECTIONS) {
            const n = this.grid.neighborOf(coord, dir, { allowOutside: false });
            if (n) result.push(n);
        }
        return result;
    }

    // --- occupancy ----------------------------------------------------------

    /** A cell can be entered if it exists and is currently empty. */
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
