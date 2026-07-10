// Voxel-grid GPU textures + grid configuration. The grid is an axis-aligned box in
// WORLD space (Z-up): origin = its min corner, dims = voxel counts per axis, cellSize =
// world units per voxel (cubic). Voxel (i,j,k) center = origin + (vec3(i,j,k)+0.5)*cellSize.
//
// Two 3D storage textures, both STORAGE_BINDING (written by the voxelize compute passes via
// textureStore) + TEXTURE_BINDING (read by later passes as a sampled texture_3d<f32>):
//   voxelRadiance rgba16float — rgb = direct-lit radiance (sun N·L·vis) + emission, a = occupancy.
//   voxelEmission rgba16float — the EMITTER class's own scatter target (single mip). Emitters and
//     occluders write different volumes so neither overwrites the other; voxelize then copies
//     voxelEmission into voxelRadiance mip 0 and the occluder scatter MERGES on top (sum rgb,
//     max a). Nothing downstream reads voxelEmission — it exists to kill the emitter/occluder
//     shared-voxel conflict that "emitter-wins" write ordering used to paper over.
//
// voxelRadiance carries a FULL MIP PYRAMID (voxelMipLevelCount levels): mip 0 is written by
// the voxelize pass, the coarser levels by the voxelMip compute pass (isotropic, opacity-
// weighted downsample). The pyramid is what the VCT cone-tracing path samples by LOD.

export type VoxelGridConfig = {
  // World-space min corner of the grid box.
  originX: number;
  originY: number;
  originZ: number;
  // Voxel counts per axis.
  dimX: number;
  dimY: number;
  dimZ: number;
  // World units per voxel (cubic).
  cellSize: number;
};

// INITIAL box (the XY origin follows the camera at runtime — createVoxelSystem.updateGridOrigin;
// only the extent = dims × cellSize and the Z range stay fixed). 256×256×64 = 4 194 304 voxels →
// voxelRadiance (rgba16float + mip pyramid) ≈ 38 MB (+ the same again for voxelEmission mip 0).
// Fine enough to resolve small objects in GI; raise cellSize via the GUI for coarser/cheaper.
export const DEFAULT_VOXEL_GRID: VoxelGridConfig = {
  originX: -32,
  originY: -32,
  originZ: -2,
  dimX: 256,
  dimY: 256,
  dimZ: 64,
  cellSize: 0.5,
};

// Number of mip levels for the voxelRadiance pyramid: full chain down to the 1-voxel mip
// along the LARGEST axis (1 + floor(log2(max dim))).
export function voxelMipLevelCount(dimX: number, dimY: number, dimZ: number): number {
  return 1 + Math.floor(Math.log2(Math.max(dimX, dimY, dimZ)));
}

export type VoxelTextures = {
  voxelRadiance: GPUTexture;
  voxelEmission: GPUTexture;
};

// ===== Screen-space probe volume (the A/B alternative to the world SH-L1 probe volume). =====
// One probe per SCREEN_PROBE_TILE×SCREEN_PROBE_TILE full-canvas-pixel tile (Lumen default
// DownsampleFactor = 16). Anchored to the VISIBLE surface (not a fixed world grid), so probe
// density follows the camera. The gather (voxelScreenProbe) writes SH-L1 into 3 rgba16float 2D
// textures + the probe's representative pixel/validity + world anchor P + world normal N (6 storage
// textures — gpu.ts requests the adapter cap). Canvas-sized-derived → recreated on resize.
export const SCREEN_PROBE_TILE = 16; // DEFAULT full-res px / screen probe (Lumen default); GUI-tunable

export type ScreenProbeGrid = { w: number; h: number };
export function screenProbeGridDims(
  canvasW: number,
  canvasH: number,
  tile: number = SCREEN_PROBE_TILE,
): ScreenProbeGrid {
  return {
    w: Math.max(1, Math.ceil(canvasW / tile)),
    h: Math.max(1, Math.ceil(canvasH / tile)),
  };
}

export type ScreenProbeTextures = {
  shR: GPUTexture;
  shG: GPUTexture;
  shB: GPUTexture;
  pix: GPUTexture;
  // Per-probe world anchor P (rgba32float, .xyz) + world normal N (rgba16float, .xyz), written by the
  // gather (which already reconstructs both for the cone origin). The resolve reads these directly —
  // ONE textureLoad each per tap — instead of re-doing normal+depth+unproject from the full-res
  // G-buffer for every probe every pixel (the P1 perf win). `pix` still carries the probe's screen
  // pixel (for the spatial kernel) + validity + footprint.
  pos: GPUTexture;
  nrm: GPUTexture;
};

// The 3 SH-L1 channel textures (.xyzw = the 4 SH-L1 coefficients) + the probe-geometry texture
// (.xy = representative full-res pixel, .z = validity). Written by the voxelScreenProbe compute
// pass (textureStore) and read by the cone pass (textureLoad — point, no filtering). 2D, single
// mip. rgba32float supports STORAGE_BINDING write in core WebGPU and is point-loaded.
//
// STAGE 3 (temporal accumulation): the system owns TWO full sets and PING-PONGS them by frame
// parity — the CURRENT set is the gather's storage-write target (and what the resolve/debug read
// this frame); the OTHER set is last frame's output = the HISTORY the gather reprojects and
// blends from. Both roles need the same usage (STORAGE_BINDING when current + TEXTURE_BINDING when
// history/read), so one create function serves both sets — no copies, the swap is pure rebinding.
// History VALIDATION needs only pos/nrm (+ the SH being blended): an invalid probe stores ZERO
// pos/nrm, so |nrm|²≈0 doubles as the history-validity test and `pix` never needs a history binding.
//
// ADDRESSING: one texel per probe, IDENTITY-mapped — the probe of tile (tx,ty) lives at texel
// (tx,ty) (slot = ty*gw + tx).
export function createScreenProbeTextures(
  device: GPUDevice,
  grid: ScreenProbeGrid,
): ScreenProbeTextures {
  const size: [number, number] = [grid.w, grid.h];
  const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
  const sh = () => device.createTexture({ size, dimension: "2d", format: "rgba16float", usage });
  const pix = device.createTexture({ size, dimension: "2d", format: "rgba32float", usage });
  // pos = world P (needs 32-bit for world-space plane-reject precision); nrm = world N (16-bit half
  // is ample for a normal-similarity weight). Both written by the gather, read point-wise by the
  // resolve → the P1 win (no per-tap G-buffer unproject).
  const pos = device.createTexture({ size, dimension: "2d", format: "rgba32float", usage });
  // shR/shG/shB = the raw gather output; the cone resolve reads them directly (unified multi-probe
  // average, no separate blur set).
  return { shR: sh(), shG: sh(), shB: sh(), pix, pos, nrm: sh() };
}

// Destroy one full atlas set (the recreate helpers own TWO sets under the Stage-3 ping-pong —
// both are destroyed + recreated together so a stale-sized history can never be sampled).
export function destroyScreenProbeTextures(t: ScreenProbeTextures) {
  t.shR.destroy();
  t.shG.destroy();
  t.shB.destroy();
  t.pix.destroy();
  t.pos.destroy();
  t.nrm.destroy();
}

// ===== Anisotropic directional voxels (the anti-leak for the far-field cone samples). =====
// Six directional radiance volumes (−X,+X,−Y,+Y,−Z,+Z). Each is HALF the iso mip-0 resolution
// (its level 0 already integrates a 2×2×2 iso block → equals the iso mip-1 resolution) and carries
// its OWN mip pyramid. voxelAnisoBase builds level 0 from the iso voxelRadiance mip 0 (accumulating
// each block FRONT-TO-BACK along that direction's axis); voxelAnisoVolume builds the coarser levels
// per-direction. The cone trace picks the 3 volumes facing the cone direction (sign of dir) and
// blends them by dir² → DIRECTION-correct occlusion: a near occluder masks the far voxel it shadows,
// which the isotropic pyramid (a single direction-agnostic average) cannot do — that is the leak
// (light bleeding through thin walls) anisotropic VCT removes.
export type AnisoTextures = {
  negX: GPUTexture;
  posX: GPUTexture;
  negY: GPUTexture;
  posY: GPUTexture;
  negZ: GPUTexture;
  posZ: GPUTexture;
};

// Aniso level-0 dims = iso mip-0 dims halved (floored at 1): level 0 integrates a 2×2×2 iso block,
// so the directional pyramid starts one step below the iso grid.
export function anisoBaseDims(dimX: number, dimY: number, dimZ: number) {
  return { x: Math.max(1, dimX >> 1), y: Math.max(1, dimY >> 1), z: Math.max(1, dimZ >> 1) };
}

// Six directional volumes at anisoBaseDims resolution, each with a full mip pyramid. Both
// STORAGE_BINDING (written by the aniso base/volume compute passes) + TEXTURE_BINDING (sampled by
// the cone pass). ~half the iso mip-0 voxel count PER direction × 6 (+ mip pyramids) → e.g. at the
// 256×256×64 default this is 6 × 128×128×32 × rgba16float × ~1.33 ≈ 33 MB.
export function createAnisoTextures(
  device: GPUDevice,
  grid: VoxelGridConfig = DEFAULT_VOXEL_GRID,
): AnisoTextures {
  const b = anisoBaseDims(grid.dimX, grid.dimY, grid.dimZ);
  const size: [number, number, number] = [b.x, b.y, b.z];
  const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
  const mipLevelCount = voxelMipLevelCount(b.x, b.y, b.z);
  const make = () =>
    device.createTexture({ size, dimension: "3d", format: "rgba16float", mipLevelCount, usage });
  return { negX: make(), posX: make(), negY: make(), posY: make(), negZ: make(), posZ: make() };
}

export function createVoxelTextures(
  device: GPUDevice,
  grid: VoxelGridConfig = DEFAULT_VOXEL_GRID,
): VoxelTextures {
  const size: [number, number, number] = [grid.dimX, grid.dimY, grid.dimZ];
  const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;

  // COPY_DST: voxelize copies voxelEmission into mip 0 (the emitter-only voxels) before the
  // occluder scatter merges on top.
  const voxelRadiance = device.createTexture({
    size,
    dimension: "3d",
    format: "rgba16float",
    mipLevelCount: voxelMipLevelCount(grid.dimX, grid.dimY, grid.dimZ),
    usage: usage | GPUTextureUsage.COPY_DST,
  });

  // Emitter scatter target (single mip — only mip 0 semantics exist for it). COPY_SRC for the
  // emission → radiance mip-0 copy; TEXTURE_BINDING for the occluder pass's merge read.
  const voxelEmission = device.createTexture({
    size,
    dimension: "3d",
    format: "rgba16float",
    usage: usage | GPUTextureUsage.COPY_SRC,
  });

  return { voxelRadiance, voxelEmission };
}
