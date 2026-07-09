// EMITTER LIGHTS sub-system (CPU clustered light cull). Owns the aimed-emitter storage buffer
// (uLights) + the clustered-cull table (uLightClusters) and their CPU scratch, and exposes
// setLights() (upload the emitters + refill/upload the cluster table each frame) plus
// recreateLightClusters() (resize the cluster table to the current grid + baked clusterDiv/clusterCap).
// Extracted verbatim from createVoxelSystem — behavior is byte-for-byte identical.
import type { VoxelBakedConfig } from "./voxelConfig.ts";
import { assignLightClusters } from "./lightClustering.ts";

export type EmitterLightsDeps = {
  device: GPUDevice;
  config: VoxelBakedConfig;
  // Current grid box (origin + cellSize + voxel dims). Read at recreateLightClusters()/setLights()
  // time, so it always reflects the latest buildGrid()/setCellSize().
  getGridDims: () => {
    dimX: number;
    dimY: number;
    dimZ: number;
    originX: number;
    originY: number;
    originZ: number;
    cellSize: number;
  };
  // Called after a lightsBuf GROW (destroy + recreate) so consumers that bind lightsBuf/clusterBuf
  // (the screen-probe gather's group 1) can rebuild their bind groups. Matches the old inline
  // buildScreenProbeGroups() call.
  onBuffersRecreated: () => void;
};

export function createEmitterLightsSystem(deps: EmitterLightsDeps) {
  const { device, config, getGridDims, onBuffersRecreated } = deps;

  // Auto-discovered emitter records the aimed cones importance-sample: interleaved
  // (x,y,z,radius, r,g,b,intensity) — two vec4 per light — in ONE storage buffer (the gather's
  // runtime-sized uLights, so the light count is UNCAPPED). Capacity is in LIGHTS, grow-doubled
  // in setLights (a growth destroys the buffer + rebuilds the probe bind groups — rare, and safe:
  // WebGPU completes in-flight work before reclaiming a destroyed buffer). Shared by BOTH gather
  // variants (one buffer, bound into each group 1).
  let lightsBufCapacity = 64;
  let lightsBuf = device.createBuffer({
    label: "vct emitter lights",
    size: lightsBufCapacity * 32, // 8 f32 per light
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  let coneLightCount = 0;
  // CLUSTERED LIGHT CULLING (Persson-style CPU assignment): the grid AABB divided into cells of
  // clusterDiv voxels per axis; setLights bins each emitter into every cell its influence sphere
  // overlaps (same 0.003 contribution cull as the shader). Layout mirrors the gather's
  // uLightClusters: (clusterCap + 1) u32 per cell — [base] = count, [base + 1 + k] = light index.
  // (Re)created by recreateLightClusters(): dims depend on the grid (buildGrid) AND the baked
  // clusterDiv/clusterCap (rebuild).
  let clusterDimX = 1;
  let clusterDimY = 1;
  let clusterDimZ = 1;
  let clusterArr = new Uint32Array(0);
  // Parallel to clusterArr's index slots: the light's estimated contribution at the CELL CENTER.
  // CPU-only (never uploaded) — drives the overflow policy: a full cell keeps its clusterCap
  // STRONGEST lights, not the first-come ones (first-come made a light vanish from the crowded
  // cells around itself while surviving in emptier far cells — light "beyond its sector but not
  // inside it").
  let clusterEstArr = new Float32Array(0);
  // Per-cell cache of the weakest kept entry (value + its slot in clusterArr), so a full cell
  // rejects a weaker light with ONE compare instead of scanning all cap entries. Re-initialized
  // lazily each frame by the first insert into the cell (counts are zeroed by fill(0)) — no
  // per-frame clear needed; a full-cell rescan happens only on an actual replacement.
  let clusterMinEst = new Float32Array(0);
  let clusterMinIdx = new Uint32Array(0);
  let clusterBuf: GPUBuffer | null = null;

  function recreateLightClusters() {
    const { dimX, dimY, dimZ } = getGridDims();
    clusterDimX = Math.max(1, Math.ceil(dimX / config.clusterDiv));
    clusterDimY = Math.max(1, Math.ceil(dimY / config.clusterDiv));
    clusterDimZ = Math.max(1, Math.ceil(dimZ / config.clusterDiv));
    const numCells = clusterDimX * clusterDimY * clusterDimZ;
    const len = numCells * (config.clusterCap + 1);
    clusterArr = new Uint32Array(len);
    clusterEstArr = new Float32Array(len);
    clusterMinEst = new Float32Array(numCells);
    clusterMinIdx = new Uint32Array(numCells);
    clusterBuf?.destroy();
    clusterBuf = device.createBuffer({
      label: "vct light clusters",
      size: len * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  // Upload the emitter data the aimed cones importance-sample. `data` = count×8 interleaved floats
  // (x,y,z,radius, r,g,b,intensity per light) — the uLights storage layout. UNCAPPED: the buffer
  // grow-doubles (destroy + recreate + rebuild the probe bind groups — rare; in-flight frames keep
  // the old buffer alive). The caller discovers these from the LightEmitter component each frame
  // (no manual light list). count=0 → pure Fibonacci fill cones. ONE shared buffer serves both
  // gather variants, so this is a single writeBuffer of the live prefix.
  function setLights(data: Float32Array, count: number) {
    if (count > lightsBufCapacity) {
      while (lightsBufCapacity < count) lightsBufCapacity *= 2;
      lightsBuf.destroy();
      lightsBuf = device.createBuffer({
        label: "vct emitter lights",
        size: lightsBufCapacity * 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      onBuffersRecreated();
    }
    if (count > 0) {
      device.queue.writeBuffer(lightsBuf, 0, data, 0, count * 8);
    }
    coneLightCount = count;

    // CLUSTERED CULL (CPU assignment, clear-and-refill each frame) → lightClustering.ts. Bins every
    // emitter into the cluster cells its influence sphere overlaps; a full cell keeps its clusterCap
    // STRONGEST lights. Then upload the refilled cluster table.
    const { originX, originY, originZ, cellSize } = getGridDims();
    assignLightClusters(
      data,
      count,
      config,
      { originX, originY, originZ, cellSize, clusterDimX, clusterDimY, clusterDimZ },
      { clusterArr, clusterEstArr, clusterMinEst, clusterMinIdx },
    );
    device.queue.writeBuffer(clusterBuf!, 0, clusterArr);
  }

  return {
    setLights,
    recreateLightClusters,
    // uploadProbeUniforms reads this into probeLightParamsArr[0].
    getLightCount: () => coneLightCount,
    // The screen-probe gather's group 1 binds these; it reads them fresh (they change on grow /
    // cluster recreate).
    get lightsBuf() {
      return lightsBuf;
    },
    get clusterBuf() {
      return clusterBuf;
    },
  };
}
