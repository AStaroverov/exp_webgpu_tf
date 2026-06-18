import { RockPartData } from "./RockParts.ts";
import { fbm } from "../../../../../../../../lib/fbm.ts";

export type RockGridOptions = {
  /**
   * Inscribed radius the rock fills. All parts stay within this circle, so the
   * rock is sized to the hex by the generator itself — no post-scaling.
   */
  radius: number;
  /** Grain: roughly the size of one stone chunk. Drives how many parts there are. */
  cellSize: number;
  noiseScale?: number;
  noiseOctaves?: number;
  /** Higher → more carved away at the rim (more ragged edge). */
  emptyThreshold?: number;
  /** ± size variation of a part as a fraction of its base size. */
  sizeJitter?: number;
  seed?: number;
};

/**
 * Build a rock that fills a circle of `radius` (so it sits inside a hex): a solid
 * core plus a noise-carved ragged rim. Parts are uniform-ish chunks of ~`cellSize`
 * packed to overlap, so the rock reads as one boulder — never a scatter of tiny
 * pebbles. Lone cells (no filled neighbour) are dropped, which is what removes the
 * stray small stones.
 */
export function generateGridRockShape(options: RockGridOptions): RockPartData[] {
  const {
    radius,
    cellSize,
    noiseScale = 0.1,
    noiseOctaves = 3,
    emptyThreshold = 0.2,
    sizeJitter = 0.2,
    seed = Math.floor(Math.random() * 1000000),
  } = options;

  // Square grid covering the rock's bounding box, centered on the origin.
  const n = Math.max(1, Math.ceil((2 * radius) / cellSize));
  const offset = (n * cellSize) / 2;
  // Part centers stay this far inside `radius` so the part body fits the circle.
  const maxCenter = radius - cellSize * 0.5;
  // Everything within the core is always solid (no internal holes / pebbles).
  const coreRadius = radius * 0.5;

  const centerOf = (col: number, row: number) => ({
    x: col * cellSize - offset + cellSize / 2,
    y: row * cellSize - offset + cellSize / 2,
  });

  // 1) Decide filled cells: solid core, noise-carved rim, clipped to the circle.
  const filled: boolean[][] = [];
  for (let row = 0; row < n; row++) {
    filled[row] = [];
    for (let col = 0; col < n; col++) {
      const c = centerOf(col, row);
      const dist = Math.hypot(c.x, c.y);
      if (dist > maxCenter) {
        filled[row][col] = false;
        continue;
      }
      if (dist <= coreRadius) {
        filled[row][col] = true;
        continue;
      }
      const density = fbm(col * noiseScale + seed * 0.001, row * noiseScale, seed, noiseOctaves);
      filled[row][col] = density >= emptyThreshold;
    }
  }

  const hasFilledNeighbour = (row: number, col: number): boolean => {
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        if (dr === 0 && dc === 0) continue;
        const r = row + dr;
        const c = col + dc;
        if (r >= 0 && r < n && c >= 0 && c < n && filled[r][c]) return true;
      }
    }
    return false;
  };

  // 2) Emit a chunk per filled cell, skipping lone cells (the "pebbles").
  const parts: RockPartData[] = [];
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      if (!filled[row][col]) continue;
      if (!hasFilledNeighbour(row, col)) continue;

      const c = centerOf(col, row);
      // Overlap neighbours (1.25×) so the mass is solid; vary size/shape lightly.
      const jitter = 1 + sizeJitter * (Math.random() * 2 - 1);
      const size = cellSize * 1.25 * jitter;
      const aspectNoise = fbm(col * noiseScale + 100, row * noiseScale + 100, 2);
      const aspect = 0.85 + aspectNoise * 0.3; // 0.85 .. 1.15
      const w = size * aspect;
      const h = size / aspect;
      const ox = (Math.random() - 0.5) * cellSize * 0.2;
      const oy = (Math.random() - 0.5) * cellSize * 0.2;
      const rotation = Math.random() * Math.PI * 2;

      parts.push([c.x + ox, c.y + oy, w, h, rotation]);
    }
  }

  return parts;
}
