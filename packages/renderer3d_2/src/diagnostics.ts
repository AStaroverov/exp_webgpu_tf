// Numeric diagnostics for the world-space Radiance Cascades (Model A: height layers).
//
// On demand (key L / GUI button) it reads back the raw gather (probeRad[c]) for every
// cascade AND the final merged c0 (probeMerge[0]), then prints ONE compact, paste-able
// numeric report — so lighting bugs are reported as numbers, not words.
//
// The probe atlases are now 2D-ARRAY textures: gridZ horizontal probe sheets stacked
// at world heights z_k = baseZ + k*cellZ. Per cascade c the atlas is
// [gx_c*dirW_c, gy_c*dirW_c, gridZ] with gx_c = max(1, gridX>>c), gy_c = max(1, gridY>>c),
// dirW_c = dir0W<<c — so each cascade can differ in width/height. We read back TWO
// representative layers per cascade: layer 0 (the base sheet) and the middle layer
// floor(gridZ/2), labeling each with its world z.
//
// Per (cascade, layer) it reports whether the gather caught the light and where:
// mean/max radiance, visibility hitFrac, the brightest probe's world XY + its single
// brightest direction cell (rgb|vis + decoded world dir). That localizes whether a bug
// is in a cascade's gather, in the merge, or just expected RC parallax.

import { WORLD_CASCADE_COUNT, WORLD_DIR0_W } from "./WGSL/createFrame.ts";

export type DiagRuntime = {
  cell0: number;
  baseZ: number; // world height of layer 0
  cellZ: number; // world units between layers
  gridX: number; // probes along x at cascade 0
  gridY: number; // probes along y at cascade 0
  gridZ: number; // number of z layers
  intervalEnd: number; // base interval (c0)
  gatherSteps: number;
  ambient: number;
  sunEnabled: boolean;
  sunAngle: number;
  sunIntensity: number;
  cameraX: number;
  cameraY: number;
  instanceCount: number;
};

// Per-cascade atlas geometry (derived once per dump from the runtime params).
type CascadeGeom = {
  dirW: number; // dir0W << c
  gx: number; // max(1, gridX >> c)
  gy: number; // max(1, gridY >> c)
  wSide: number; // gx * dirW  (atlas width)
  hSide: number; // gy * dirW  (atlas height)
  cell: number; // cell0 * 2^c
};

function cascadeGeom(rt: DiagRuntime, c: number): CascadeGeom {
  const dirW = WORLD_DIR0_W << c;
  const gx = Math.max(1, rt.gridX >> c);
  const gy = Math.max(1, rt.gridY >> c);
  return {
    dirW,
    gx,
    gy,
    wSide: gx * dirW,
    hSide: gy * dirW,
    cell: rt.cell0 * 2 ** c,
  };
}

export function createWorldRCDiagnostics({
  device,
  getProbeRad,
  getProbeMerge,
  getWorldLit,
  getRuntime,
}: {
  device: GPUDevice;
  getProbeRad: () => GPUTexture[]; // probeRad[c], length N, each [wSide_c, hSide_c, gridZ]
  getProbeMerge: () => GPUTexture[]; // probeMerge[c], length N-1, each [wSide_c, hSide_c, gridZ]
  getWorldLit: () => GPUTexture;
  getRuntime: () => DiagRuntime;
}) {
  let busy = false;

  async function dump() {
    if (busy) {
      console.warn("[world-rc-diag] still reading the previous dump, skipping");
      return;
    }
    busy = true;
    try {
      const rt = getRuntime();
      const rad = getProbeRad();
      const merge = getProbeMerge();

      // Two representative layers: base (0) and the middle of the stack.
      const layer0 = 0;
      const layerMid = Math.floor(rt.gridZ / 2);
      const layers = layer0 === layerMid ? [layer0] : [layer0, layerMid];

      // radData[layerIndex][c] — read back each cascade atlas at each chosen layer.
      const radData = await Promise.all(
        layers.map((k) =>
          Promise.all(
            rad.map((t, c) => {
              const g = cascadeGeom(rt, c);
              return readTextureRGBA(device, t, g.wSide, g.hSide, "rgba16f", k);
            }),
          ),
        ),
      );

      const g0 = cascadeGeom(rt, 0);
      // merge0 is read at the middle layer (where the composite's trilinear-z most
      // likely lands for a mid-height receiver).
      const merge0 = await readTextureRGBA(device, merge[0], g0.wSide, g0.hSide, "rgba16f", layerMid);

      const lit = getWorldLit();
      const litData = await readTextureRGBA(device, lit, lit.width, lit.height, "bgra8", 0);

      console.log(
        buildReport(rt, layers, radData, layerMid, merge0, lit.width, lit.height, litData),
      );
    } catch (err) {
      console.error("[world-rc-diag] dump failed:", err);
    } finally {
      busy = false;
    }
  }

  return { dump };
}

// ---- report ---------------------------------------------------------------

function layerZ(rt: DiagRuntime, k: number): number {
  return rt.baseZ + k * rt.cellZ;
}

function buildReport(
  rt: DiagRuntime,
  layers: number[],
  rad: Float32Array[][], // [layerIndex][c]
  mergeLayer: number,
  merge0: Float32Array,
  litW: number,
  litH: number,
  lit: Float32Array,
): string {
  const L: string[] = [];
  L.push("=== WORLD-RC DIAG (height-layers, per-cascade) ===");
  L.push(
    `params: cell0=${rt.cell0} baseZ=${rt.baseZ} cellZ=${rt.cellZ} ` +
      `grid=${rt.gridX}x${rt.gridY}x${rt.gridZ} baseInterval=${rt.intervalEnd} ` +
      `steps=${rt.gatherSteps} ambient=${rt.ambient}`,
  );
  L.push(
    `layers z: [${layers.map((k) => `k${k}@z=${f2(layerZ(rt, k))}`).join(", ")}] ` +
      `(of ${rt.gridZ} total)`,
  );
  L.push(`sun: enabled=${rt.sunEnabled ? 1 : 0} angle=${f2(rt.sunAngle)} intensity=${rt.sunIntensity}`);
  const g0 = cascadeGeom(rt, 0);
  L.push(
    `camera=(${f2(rt.cameraX)},${f2(rt.cameraY)}) instances=${rt.instanceCount} ` +
      `atlas0=${g0.wSide}x${g0.hSide}x${rt.gridZ} cascades=${WORLD_CASCADE_COUNT}`,
  );

  // ---- per-cascade gather (probeRad[c]) at each representative layer ----
  L.push(
    "gather per cascade [c@layer: dirW gx gy cell interval | meanRGB max hitFrac | brightest]",
  );
  for (let li = 0; li < layers.length; li++) {
    const k = layers[li];
    const cascades = rad[li];
    for (let c = 0; c < cascades.length; c++) {
      const g = cascadeGeom(rt, c);
      const end = rt.intervalEnd * 4 ** c;
      const start = c === 0 ? 0 : rt.intervalEnd * 4 ** (c - 1);
      const ox = Math.floor(rt.cameraX / g.cell) * g.cell;
      const oy = Math.floor(rt.cameraY / g.cell) * g.cell;
      const st = channelStats(cascades[c]);
      const b = brightestProbe(cascades[c], g.wSide, g.gx, g.gy, g.dirW);
      const wx = ox + (b.i + 0.5 - g.gx * 0.5) * g.cell;
      const wy = oy + (b.j + 0.5 - g.gy * 0.5) * g.cell;
      const bc = brightestCell(cascades[c], g.wSide, b.i, b.j, g.dirW);
      L.push(
        `  c${c}@k${k}(z=${f2(layerZ(rt, k))}): dirW=${g.dirW} gx=${g.gx} gy=${g.gy} ` +
          `cell=${f2(g.cell)} iv=[${f1(start)},${f1(end)}] | ` +
          `mean=(${f4(st.mean[0])},${f4(st.mean[1])},${f4(st.mean[2])}) max=${f2(st.max[0])} ` +
          `hit=${f3(st.hitFrac)} | bright ij=(${b.i},${b.j}) world=(${f2(wx)},${f2(wy)}) ` +
          `cell#${bc.n} dir=(${f2(bc.dir[0])},${f2(bc.dir[1])},${f2(bc.dir[2])}) ` +
          `rgb=(${f2(bc.r)},${f2(bc.g)},${f2(bc.b)}|${f1(bc.vis)})`,
      );
    }
  }

  // ---- final merged c0 (what the composite reads), at the middle layer ----
  const ms = channelStats(merge0);
  L.push(
    `merge0 @k${mergeLayer}(z=${f2(layerZ(rt, mergeLayer))}) ` +
      `mean=(${f4(ms.mean[0])},${f4(ms.mean[1])},${f4(ms.mean[2])}) max=${f2(ms.max[0])} ` +
      `nan=${ms.nan} inf=${ms.inf} zeroRgbFrac=${f3(ms.zeroRgbFrac)}`,
  );
  const cx = g0.gx >> 1;
  const cy = g0.gy >> 1;
  L.push(
    `merge0 center probe ij=(${cx},${cy}) world=(0,0): ` +
      tileDump(merge0, g0.wSide, cx, cy, WORLD_DIR0_W),
  );
  const mb = brightestProbe(merge0, g0.wSide, g0.gx, g0.gy, WORLD_DIR0_W);
  L.push(
    `merge0 bright probe ij=(${mb.i},${mb.j}): ` +
      tileDump(merge0, g0.wSide, mb.i, mb.j, WORLD_DIR0_W),
  );

  // ---- final composite ----
  const ls = channelStats(lit);
  L.push(
    `lit ${litW}x${litH} R[max=${f3(ls.max[0])} mean=${f4(ls.mean[0])}] ` +
      `G[mean=${f4(ls.mean[1])}] B[mean=${f4(ls.mean[2])}] litFrac=${f3(1 - ls.zeroRgbFrac)}`,
  );
  return L.join("\n");
}

// One small octahedral tile as `d0:(r,g,b|vis) ...` (only sane for small dirW).
function tileDump(atlas: Float32Array, wSide: number, i: number, j: number, dirW: number): string {
  const out: string[] = [];
  let n = 0;
  for (let v = 0; v < dirW; v++) {
    for (let u = 0; u < dirW; u++) {
      const o = ((j * dirW + v) * wSide + (i * dirW + u)) * 4;
      out.push(`d${n}:(${f2(atlas[o])},${f2(atlas[o + 1])},${f2(atlas[o + 2])}|${f1(atlas[o + 3])})`);
      n++;
    }
  }
  return out.join(" ");
}

function brightestProbe(
  atlas: Float32Array,
  wSide: number,
  gridX: number,
  gridY: number,
  dirW: number,
) {
  let best = { i: 0, j: 0, mean: -1 };
  for (let j = 0; j < gridY; j++) {
    for (let i = 0; i < gridX; i++) {
      let s = 0;
      for (let v = 0; v < dirW; v++) {
        for (let u = 0; u < dirW; u++) {
          const o = ((j * dirW + v) * wSide + (i * dirW + u)) * 4;
          s += atlas[o] + atlas[o + 1] + atlas[o + 2];
        }
      }
      const mean = s / (dirW * dirW * 3);
      if (mean > best.mean) best = { i, j, mean };
    }
  }
  return best;
}

// Brightest direction cell within a probe tile + its decoded world direction.
function brightestCell(atlas: Float32Array, wSide: number, i: number, j: number, dirW: number) {
  let best = { n: 0, u: 0, v: 0, s: -1 };
  let n = 0;
  for (let v = 0; v < dirW; v++) {
    for (let u = 0; u < dirW; u++) {
      const o = ((j * dirW + v) * wSide + (i * dirW + u)) * 4;
      const s = atlas[o] + atlas[o + 1] + atlas[o + 2];
      if (s > best.s) best = { n, u, v, s };
      n++;
    }
  }
  const o = ((j * dirW + best.v) * wSide + (i * dirW + best.u)) * 4;
  const ex = ((best.u + 0.5) / dirW) * 2 - 1;
  const ey = ((best.v + 0.5) / dirW) * 2 - 1;
  return {
    n: best.n,
    dir: octDecode(ex, ey),
    r: atlas[o],
    g: atlas[o + 1],
    b: atlas[o + 2],
    vis: atlas[o + 3],
  };
}

// Matches the WGSL oct_decode.
function octDecode(ex: number, ey: number): [number, number, number] {
  let x = ex;
  let y = ey;
  let z = 1 - Math.abs(ex) - Math.abs(ey);
  if (z < 0) {
    const nx = (1 - Math.abs(y)) * Math.sign(x);
    const ny = (1 - Math.abs(x)) * Math.sign(y);
    x = nx;
    y = ny;
  }
  const len = Math.hypot(x, y, z) || 1;
  return [x / len, y / len, z / len];
}

function channelStats(data: Float32Array) {
  const min = [Infinity, Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity, -Infinity];
  const sum = [0, 0, 0, 0];
  let nan = 0;
  let inf = 0;
  let zeroRgb = 0;
  let hit = 0;
  const px = data.length / 4;
  for (let p = 0; p < px; p++) {
    const o = p * 4;
    for (let c = 0; c < 4; c++) {
      const x = data[o + c];
      if (Number.isNaN(x)) {
        nan++;
        continue;
      }
      if (!Number.isFinite(x)) {
        inf++;
        continue;
      }
      if (x < min[c]) min[c] = x;
      if (x > max[c]) max[c] = x;
      sum[c] += x;
    }
    if (data[o] === 0 && data[o + 1] === 0 && data[o + 2] === 0) zeroRgb++;
    if (data[o + 3] < 0.5) hit++;
  }
  return {
    min,
    max,
    mean: sum.map((s) => s / px),
    nan,
    inf,
    zeroRgbFrac: zeroRgb / px,
    hitFrac: hit / px,
  };
}

// ---- GPU texture readback -------------------------------------------------

async function readTextureRGBA(
  device: GPUDevice,
  tex: GPUTexture,
  w: number,
  h: number,
  kind: "rgba16f" | "bgra8",
  layer: number,
): Promise<Float32Array> {
  const bytesPerPixel = kind === "rgba16f" ? 8 : 4;
  const unpadded = w * bytesPerPixel;
  const bytesPerRow = Math.ceil(unpadded / 256) * 256;

  const buffer = device.createBuffer({
    size: bytesPerRow * h,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  // origin.z selects the array layer of a 2D-array texture; depthOrArrayLayers:1 copies
  // exactly that one layer.
  encoder.copyTextureToBuffer(
    { texture: tex, origin: { x: 0, y: 0, z: layer } },
    { buffer, bytesPerRow, rowsPerImage: h },
    { width: w, height: h, depthOrArrayLayers: 1 },
  );
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);

  const raw = buffer.getMappedRange();
  const out = new Float32Array(w * h * 4);
  if (kind === "rgba16f") {
    const u16 = new Uint16Array(raw);
    const rowU16 = bytesPerRow / 2;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const src = y * rowU16 + x * 4;
        const dst = (y * w + x) * 4;
        out[dst] = half2float(u16[src]);
        out[dst + 1] = half2float(u16[src + 1]);
        out[dst + 2] = half2float(u16[src + 2]);
        out[dst + 3] = half2float(u16[src + 3]);
      }
    }
  } else {
    const u8 = new Uint8Array(raw);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const src = y * bytesPerRow + x * 4;
        const dst = (y * w + x) * 4;
        out[dst] = u8[src + 2] / 255;
        out[dst + 1] = u8[src + 1] / 255;
        out[dst + 2] = u8[src] / 255;
        out[dst + 3] = u8[src + 3] / 255;
      }
    }
  }
  buffer.unmap();
  buffer.destroy();
  return out;
}

function half2float(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  const sign = s ? -1 : 1;
  if (e === 0) return sign * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : sign * Infinity;
  return sign * Math.pow(2, e - 15) * (1 + f / 1024);
}

// ---- number formatting ----------------------------------------------------
function f1(x: number) {
  return Number.isFinite(x) ? x.toFixed(1) : String(x);
}
function f2(x: number) {
  return Number.isFinite(x) ? x.toFixed(2) : String(x);
}
function f3(x: number) {
  return Number.isFinite(x) ? x.toFixed(3) : String(x);
}
function f4(x: number) {
  return Number.isFinite(x) ? x.toFixed(4) : String(x);
}
