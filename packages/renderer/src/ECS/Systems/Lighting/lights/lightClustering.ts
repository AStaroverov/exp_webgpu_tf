import type { VoxelBakedConfig } from "../core/voxelConfig.ts";

// CPU clustered light culling (Persson-style, clear-and-refill each frame). Bins every emitter into
// the cluster cells its influence sphere overlaps; a full cell keeps its clusterCap STRONGEST lights
// (estimated contribution at the cell center). Pure: reads the emitter records + grid params, writes
// the preallocated cluster scratch (which setLights then uploads to the GPU).

export type ClusterGrid = {
  cellSize: number;
  originX: number;
  originY: number;
  originZ: number;
  clusterDimX: number;
  clusterDimY: number;
  clusterDimZ: number;
};

// clusterArr layout mirrors the gather's uLightClusters: (clusterCap + 1) u32 per cell —
// [base] = count, [base + 1 + k] = light index. clusterEstArr is CPU-only (parallel to clusterArr's
// index slots) and holds each entry's estimated contribution; clusterMinEst/clusterMinIdx cache the
// weakest kept entry per cell so a full cell rejects a weaker light in O(1).
export type ClusterScratch = {
  clusterArr: Uint32Array;
  clusterEstArr: Float32Array;
  clusterMinEst: Float32Array;
  clusterMinIdx: Uint32Array;
};

// `data` = count×8 interleaved emitter floats (x,y,z,radius, r,g,b,intensity). Influence radius =
// where the shader's own 0.003 contribution gate would cull the light: atten = 1/(1 + F·d²/lr²) ≥
// 0.003 / (maxLum·direct) → R = lr·√((maxLum·direct/0.003 − 1)/F) — the SAME radius the shader's
// range window scales the light to exactly zero at (no boundary step). emitterFalloff = 0 ⇒ R = ∞,
// and the ±Infinity arithmetic clamps to the whole grid without a special case.
export function assignLightClusters(
  data: Float32Array,
  count: number,
  config: VoxelBakedConfig,
  grid: ClusterGrid,
  scratch: ClusterScratch,
): void {
  const { originX, originY, originZ, cellSize, clusterDimX, clusterDimY, clusterDimZ } = grid;
  const { clusterArr, clusterEstArr, clusterMinEst, clusterMinIdx } = scratch;
  clusterArr.fill(0);
  const cw = cellSize * config.clusterDiv; // cluster cell size, world units
  const cap = config.clusterCap;
  const stride = cap + 1;
  for (let i = 0; i < count; i++) {
    const o = i * 8;
    const maxLum =
      Math.max(data[o + 4], data[o + 5], data[o + 6]) *
      Math.abs(data[o + 7]) *
      config.emitterDirect;
    if (maxLum < 0.003) continue; // the shader would cull it in every cell
    const lx = data[o + 0];
    const ly = data[o + 1];
    const lz = data[o + 2];
    const lr = Math.max(data[o + 3], 1e-3);
    const F = config.emitterFalloff;
    const R = F > 0 ? lr * Math.sqrt((maxLum / 0.003 - 1) / F) : Infinity;
    const x0 = Math.max(0, Math.floor((lx - R - originX) / cw));
    const x1 = Math.min(clusterDimX - 1, Math.floor((lx + R - originX) / cw));
    const y0 = Math.max(0, Math.floor((ly - R - originY) / cw));
    const y1 = Math.min(clusterDimY - 1, Math.floor((ly + R - originY) / cw));
    const z0 = Math.max(0, Math.floor((lz - R - originZ) / cw));
    const z1 = Math.min(clusterDimZ - 1, Math.floor((lz + R - originZ) / cw));
    const kAtt = F / (lr * lr); // est = maxLum / (1 + kAtt·d²)
    const dx0 = lx - (originX + (x0 + 0.5) * cw);
    for (let z = z0; z <= z1; z++) {
      const dz = lz - (originZ + (z + 0.5) * cw);
      const dz2 = dz * dz;
      for (let y = y0; y <= y1; y++) {
        const dy = ly - (originY + (y + 0.5) * cw);
        const dyz2 = dz2 + dy * dy;
        let cell = (z * clusterDimY + y) * clusterDimX + x0;
        let base = cell * stride;
        let dx = dx0;
        for (let x = x0; x <= x1; x++, cell++, base += stride, dx -= cw) {
          // Contribution estimate at the cell center (the same falloff the shader applies) —
          // the cell's keep/replace ranking key.
          const est = maxLum / (1 + kAtt * (dx * dx + dyz2));
          // The sphere-AABB corners: est here is below the shader's cull gate, so the cell
          // would receive exactly zero from this light — skip it.
          if (est < 0.003) continue;
          const c = clusterArr[base];
          if (c < cap) {
            const slot = base + 1 + c;
            clusterArr[slot] = i;
            clusterEstArr[slot] = est;
            clusterArr[base] = c + 1;
            if (c === 0 || est < clusterMinEst[cell]) {
              clusterMinEst[cell] = est;
              clusterMinIdx[cell] = slot;
            }
          } else if (est > clusterMinEst[cell]) {
            // Full cell: replace the current weakest entry (cached), then rescan the cap
            // entries ONLY here — the common weaker-light case is the O(1) reject above.
            const wk = clusterMinIdx[cell];
            clusterArr[wk] = i;
            clusterEstArr[wk] = est;
            let mn = base + 1;
            for (let k = base + 2; k < base + 1 + cap; k++) {
              if (clusterEstArr[k] < clusterEstArr[mn]) mn = k;
            }
            clusterMinEst[cell] = clusterEstArr[mn];
            clusterMinIdx[cell] = mn;
          }
        }
      }
    }
  }
}
