// Voxel-grid GPU textures + grid configuration. The grid is an axis-aligned box in
// WORLD space (Z-up): origin = its min corner, dims = voxel counts per axis, cellSize =
// world units per voxel (cubic). Voxel (i,j,k) center = origin + (vec3(i,j,k)+0.5)*cellSize.
//
// One 3D storage texture, both STORAGE_BINDING (written by the voxelize compute pass via
// textureStore) + TEXTURE_BINDING (read by later passes as a sampled texture_3d<f32>):
//   voxelRadiance rgba16float — rgb = direct-lit radiance (sun N·L·vis) + emission, a = occupancy.
//
// voxelRadiance carries a FULL MIP PYRAMID (voxelMipLevelCount levels): mip 0 is written by
// the voxelize pass, the coarser levels by the voxelMip compute pass (isotropic, opacity-
// weighted downsample). The pyramid is what the VCT cone-tracing path samples by LOD.
//
// (The former voxelAlbedo / voxelEmission 3D volumes were removed: their only reader was the
// unimported voxelTrace.wgsl.ts; the composite reads the 2D G-buffer emission, not a volume.)

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

// World box x,y in [-32, 32], z in [-2, 14] (extent 64×64×16). At cellSize 0.25 that is
// 256×256×64 = 4 194 304 voxels → voxelRadiance (rgba16float + mip pyramid) ≈ 38 MB. Fine enough
// to resolve small objects in GI; raise cellSize via the GUI for coarser/cheaper, lower for finer.
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
// parity — the CURRENT set is the gather's storage-write target (and what refine/resolve/debug
// read this frame); the OTHER set is last frame's output = the HISTORY the gather reprojects and
// blends from. Both roles need the same usage (STORAGE_BINDING when current + TEXTURE_BINDING when
// history/read), so one create function serves both sets — no copies, the swap is pure rebinding.
// History VALIDATION needs only pos/nrm (+ the SH being blended): an invalid probe stores ZERO
// pos/nrm, so |nrm|²≈0 doubles as the history-validity test and `pix` never needs a history binding.
//
// FLAT-ATLAS ADDRESSING (slot → texel). The 4 textures are a flat probe atlas, not a literal
// screen grid. Atlas WIDTH in probes = gw = grid.w; a probe's global SLOT maps to atlas texel
// (slot % gw, slot / gw). The UNIFORM block occupies rows [0, gh): a uniform probe for tile
// (tx,ty) has slot = ty*gw + tx → texel (tx,ty) — an IDENTITY mapping (no indirection). An
// ADAPTIVE block (Phase 1+) occupies rows [gh, gh+adaptiveRows) for extra sub-tile probes.
// Phase 0 passes adaptiveRows = 0, so the atlas is exactly [gw, gh] as before and every texel
// is unchanged.
export function createScreenProbeTextures(
  device: GPUDevice,
  grid: ScreenProbeGrid,
  adaptiveRows: number = 0,
): ScreenProbeTextures {
  const size: [number, number] = [grid.w, grid.h + adaptiveRows];
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

// ===== Adaptive screen-probe atlas sizing (single 16→8 level, light-adaptive). =====
// K = max adaptive probes per coarse (16px) tile = the fixed stride of tileIndices. A 16→8 split
// yields at most 3 useful children, but a finer live cell divisor (GUI-tunable, e.g. 16→4/16→1) can
// request many more per tile, so K = 8 gives headroom (the surplus beyond K is allocated but dropped
// from the per-tile list). Shared between the buffer sizing (tileIndices length) and the
// refine/resolve/debug shaders (interpolated as a WGSL const), so the CPU stride and the GPU stride
// can never disagree.
export const SCREEN_PROBE_K = 8;

// Atlas + buffer capacity from the coarse grid + the adaptive budget fraction. numUniform = the
// uniform block (one probe per 16px tile, identity-mapped into atlas rows [0, gh)); maxAdaptive =
// numUniform * adaptiveFraction extra probes (Phase-1 default 1.0 → the atlas is 2× uniform, higher
// than Lumen's ~0.5 because we amortize NOTHING across frames — every probe is paid fresh). The
// adaptive block lands in atlas rows [gh, gh + adaptiveRows).
export type ScreenProbeCounts = {
  numUniform: number;
  maxAdaptive: number;
  adaptiveRows: number;
};
export function screenProbeCounts(
  grid: ScreenProbeGrid,
  adaptiveFraction: number,
): ScreenProbeCounts {
  const numUniform = grid.w * grid.h;
  const maxAdaptive = Math.max(0, Math.round(numUniform * Math.max(0, adaptiveFraction)));
  const adaptiveRows = Math.ceil(maxAdaptive / grid.w);
  return { numUniform, maxAdaptive, adaptiveRows };
}

// The indirection + allocator STORAGE BUFFERS (NOT textures — WebGPU forbids storage-texture
// atomics, and buffers stay off the 4-storage-texture cap the gather pass lives under). These are
// SHARED across the classify / refine / args / gather / resolve passes, so they are RAW
// device.createBuffer()s bound MANUALLY at each shader's declared binding (the argsBuf / passBuf
// pattern) — never routed through GPUVariable.getGPUBuffer (which cannot express INDIRECT usage and
// would hand each shader its OWN buffer instead of the one shared instance).
export type ScreenProbeBuffers = {
  // atomic<u32> ×2: [0] = bump-allocator counter for adaptive probes, [1] = sticky "budget
  // exceeded" flag (set when an atomicAdd returns >= maxAdaptive). Cleared each frame.
  counter: GPUBuffer;
  // array<vec4<u32>>, numUniform + maxAdaptive: per-probe record (packed repr pixel + level),
  // indexed by GLOBAL slot (uniform slots 0..numUniform-1, adaptive numUniform..).
  data: GPUBuffer;
  // array<atomic<u32>>, numUniform: count of adaptive probes whose parent is that coarse tile.
  header: GPUBuffer;
  // array<u32>, numUniform*K: fixed-stride list — [tileIdx*K + j] = the global slot of the tile's
  // j-th adaptive probe (only the first min(header, K) entries per tile are meaningful).
  indices: GPUBuffer;
  // array<u32,3> = dispatchWorkgroupsIndirect args [ceil(total/GATHER_WG), 1, 1] for the gather.
  // RAW STORAGE|INDIRECT|COPY_DST buffer (COPY_SRC too for diagnostics) — the one usage GPUVariable
  // cannot emit.
  args: GPUBuffer;
};
export function createScreenProbeBuffers(
  device: GPUDevice,
  counts: ScreenProbeCounts,
): ScreenProbeBuffers {
  // COPY_SRC on every buffer so the throttled budget readback (counter) and any future diagnostics
  // can copy them to a staging buffer; COPY_DST so the per-frame clearBuffer works.
  const storageUsage =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const total = counts.numUniform + counts.maxAdaptive;
  return {
    counter: device.createBuffer({ size: 2 * 4, usage: storageUsage }),
    data: device.createBuffer({ size: Math.max(1, total) * 16, usage: storageUsage }),
    header: device.createBuffer({ size: Math.max(1, counts.numUniform) * 4, usage: storageUsage }),
    indices: device.createBuffer({
      size: Math.max(1, counts.numUniform * SCREEN_PROBE_K) * 4,
      usage: storageUsage,
    }),
    args: device.createBuffer({
      size: 3 * 4,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.INDIRECT |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    }),
  };
}

export function destroyScreenProbeBuffers(b: ScreenProbeBuffers) {
  b.counter.destroy();
  b.data.destroy();
  b.header.destroy();
  b.indices.destroy();
  b.args.destroy();
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

  const voxelRadiance = device.createTexture({
    size,
    dimension: "3d",
    format: "rgba16float",
    mipLevelCount: voxelMipLevelCount(grid.dimX, grid.dimY, grid.dimZ),
    usage,
  });

  return { voxelRadiance };
}
