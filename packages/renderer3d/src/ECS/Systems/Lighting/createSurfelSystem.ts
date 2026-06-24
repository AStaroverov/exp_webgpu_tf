import { mat4 } from "gl-matrix";
import { GPUShader } from "../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../Shader/index.ts";
import { viewProjMatrix } from "../ResizeSystem.ts";
import { shaderMeta as spawnMeta } from "./surfelSpawn.shader.ts";
import { shaderMeta as debugMeta } from "./surfelDebug.shader.ts";
import { shaderMeta as insertMeta } from "./surfelInsert.shader.ts";
import { shaderMeta as recycleMeta } from "./surfelRecycle.shader.ts";
import { createSurfelResources, SURFEL_CAP, GRID_CAP, CELL_K } from "./surfelResources.ts";

// Surfel Radiance Cascades — Stage B SYSTEM.
//
// Owns the standalone surfel buffers + grid (surfelResources) + the per-frame GPU
// passes. run(encoder, frameIndex) sequences them in command order:
//   1. clearGrid  — encoder.clearBuffer(grid) — zero the hash grid for this frame.
//   2. insert     — a COMPUTE pass (ceil(CAP/64) threads) that rebuilds the spatial
//        hash grid from the CURRENT live surfels (last frame's survivors): each live
//        surfel writes its id into every bucket its radius-disc overlaps.
//   3. spawn      — a COMPUTE pass over the G-buffer that COVERAGE-GATES allocation:
//        a pixel only allocates a surfel if its world point is NOT already covered by
//        a nearby grid surfel; the alloc itself stays gated by a low random chance.
//        World pos reconstructed from reverse-Z depth + inverse VP.
//   4. recycle    — a COMPUTE pass (ceil(CAP/64) threads) that ages each surfel's
//        recycle marker and, by a frame-seeded random heuristic (stale OR over-covered
//        by neighbors), pushes dead ids back onto the free stack.
// Together (coverage-gated spawn + recycle) the live count settles FAR below CAP and
// tracks the view (Stage A's probability-only spawn pegged at CAP).
//
// All compute pipelines (spawn/insert/recycle) use autoLayout: their groups are
// {0 (uniforms[+textures]), 2 (StorageWrite/Read surfel buffers)} — NON-contiguous,
// so autoLayout (WGSL @group reflection) is mandatory; the explicit pipeline-layout
// path would mis-pack {0,2} -> {0,1}. The standalone buffers are bound MANUALLY
// against pipeline.getBindGroupLayout(2).
//   drawDebug(pass) — an INSTANCED render (CAP instances × 6 verts) of camera-facing
//     billboard dots, drawn as an OVERLAY into the presented texture (caller opens the
//     pass with loadOp "load", no depth). Explicit pipeline layout is fine here: its
//     groups {0 (uniforms), 1 (StorageRead surfel buffers)} are CONTIGUOUS.
//
// The surfel buffers carry atomics / sized arrays the JS type-size parser can't read,
// so they are STANDALONE device buffers (surfelResources). Their VariableMetas in the
// shaders exist ONLY for WGSL emission + the (kind-based, size-agnostic) layout entry;
// we bind them MANUALLY ({ binding, resource: { buffer } }) against the reflected
// bind-group layouts — never through GPUVariable.getGPUBuffer.

export type SurfelParams = {
  spawnChance: number; // per-pixel per-frame spawn probability (low => gradual fill)
  surfelRadius: number; // FIXED world-unit radius (ortho => no depth scaling). DENSITY scale.
  // Spawn-coverage distance as a MULTIPLE of radius (1.0 = "covered within r"). The
  // single density-shape knob: smaller => denser packing, larger => sparser. cellSize,
  // the spawn coverageThreshold and the recycle threshold are ALL derived from this +
  // radius (see deriveCoverage), so they can't be desynced. (>1 needs slightly larger
  // cellSize; we clamp cellSize >= 2r so the grid still finds neighbors.)
  coverage: number;
  markerDecay: number; // per-frame decrement of a surfel's recycle marker (norw.w)
  quadPx: number; // debug dot diameter in screen pixels (zoom-independent)
};

export const DEFAULT_SURFEL_PARAMS: SurfelParams = {
  spawnChance: 0.2,
  surfelRadius: 0.2,
  coverage: 2.0,
  markerDecay: 0.125,
  quadPx: 6,
};

// Recycle distance as a fraction of the spawn-coverage distance. < 1 creates a
// HYSTERESIS dead-band [HYSTERESIS*cov, cov]·r where surfels neither spawn nor recycle
// → stable spacing, no per-frame churn at the boundary. (Inverse of the old bug where
// recycleCoverage was LOOSER than coverage, recycling freshly-spawned surfels.)
const RECYCLE_HYSTERESIS = 0.7;

// Derive the grid + spawn + recycle thresholds from radius + coverage so the three
// can never be set inconsistently. point_coverage is nor_dist = dist²/r² - 1, so a
// "covered within (k·r)" rule is nor_dist < k² - 1.
function deriveCoverage(radius: number, coverage: number) {
  return {
    // cellSize = 2r so the single-cell coverage query EXACTLY finds every surfel
    // within r of the point (insert registers a surfel in all cells its r-disc
    // overlaps). Exact for coverage <= 1; for coverage > 1 the query can miss
    // neighbors in [r, coverage·r] (single-cell limit) → slightly denser than the
    // label — fine for the intended <=1 (denser) range.
    cellSize: 2 * radius,
    coverageThreshold: coverage * coverage - 1, // spawn: covered within coverage·r
    recycleCoverage: (RECYCLE_HYSTERESIS * coverage) ** 2 - 1, // recycle: only if much closer
  };
}

export function createSurfelSystem({
  device,
  depthTexture,
  normalTexture,
  params,
}: {
  device: GPUDevice;
  depthTexture: GPUTexture;
  normalTexture: GPUTexture;
  params?: Partial<SurfelParams>;
}) {
  const p = { ...DEFAULT_SURFEL_PARAMS, ...params };

  // Live G-buffer texture refs (swapped on resize via recreate()).
  let depthTex = depthTexture;
  let normalTex = normalTexture;

  const resources = createSurfelResources(device);
  const invViewProj = mat4.create();

  // ===== SPAWN compute pipeline (autoLayout: groups {0, 2} non-contiguous) =====
  const spawnShader = new GPUShader(spawnMeta);
  const spawnPipeline = spawnShader.getComputePipeline(device, "main", { autoLayout: true });

  // Resolve the @binding numbers from the metas so manual entries stay in sync with WGSL.
  const depthBinding = spawnMeta.uniforms.depthTexture.binding;
  const normalBinding = spawnMeta.uniforms.normalTexture.binding;

  // Group 2 = the standalone surfel buffers + the hash grid (bindings 0/1/2/3, fixed
  // in the meta). Built ONCE — the buffers never change identity across resize. Spawn
  // reads the grid (StorageRead) for the coverage test before alloc.
  const spawnGroup2 = device.createBindGroup({
    layout: spawnPipeline.getBindGroupLayout(2),
    entries: [
      { binding: spawnMeta.uniforms.surfelStack.binding, resource: { buffer: resources.stack } },
      { binding: spawnMeta.uniforms.surfelPosr.binding, resource: { buffer: resources.posr } },
      { binding: spawnMeta.uniforms.surfelNorw.binding, resource: { buffer: resources.norw } },
      { binding: spawnMeta.uniforms.surfelGrid.binding, resource: { buffer: resources.grid } },
      { binding: spawnMeta.uniforms.surfelClaim.binding, resource: { buffer: resources.claim } },
    ],
  });

  // ===== INSERT compute pipeline (autoLayout: groups {0 uniform, 2 storage}) =====
  // One thread per surfel slot; rebuilds the hash grid from the current live set.
  const insertShader = new GPUShader(insertMeta);
  const insertPipeline = insertShader.getComputePipeline(device, "main", { autoLayout: true });
  // Group 0 = the single `params` uniform (cellSize, cap, gridCap, cellK).
  const insertGroup0 = device.createBindGroup({
    layout: insertPipeline.getBindGroupLayout(0),
    entries: [insertShader.uniforms.params.getBindGroupEntry(device)],
  });
  // Group 2 = posr (read) + grid (read_write). Manual; built ONCE.
  const insertGroup2 = device.createBindGroup({
    layout: insertPipeline.getBindGroupLayout(2),
    entries: [
      { binding: insertMeta.uniforms.surfelPosr.binding, resource: { buffer: resources.posr } },
      { binding: insertMeta.uniforms.surfelGrid.binding, resource: { buffer: resources.grid } },
    ],
  });

  // ===== RECYCLE compute pipeline (autoLayout: groups {0 uniforms, 2 storage}) =====
  // One thread per surfel slot; ages markers + pushes dead ids back onto the stack.
  const recycleShader = new GPUShader(recycleMeta);
  const recyclePipeline = recycleShader.getComputePipeline(device, "main", { autoLayout: true });
  // Group 0 = params (cellSize, gridCap, cellK, frameIndex) + params2 (cap, markerDecay,
  // recycleCoverage, _).
  const recycleGroup0 = device.createBindGroup({
    layout: recyclePipeline.getBindGroupLayout(0),
    entries: [
      recycleShader.uniforms.params.getBindGroupEntry(device),
      recycleShader.uniforms.params2.getBindGroupEntry(device),
    ],
  });
  // Group 2 = stack (rw) + posr (rw) + norw (rw) + grid (read). Manual; built ONCE.
  const recycleGroup2 = device.createBindGroup({
    layout: recyclePipeline.getBindGroupLayout(2),
    entries: [
      { binding: recycleMeta.uniforms.surfelStack.binding, resource: { buffer: resources.stack } },
      { binding: recycleMeta.uniforms.surfelPosr.binding, resource: { buffer: resources.posr } },
      { binding: recycleMeta.uniforms.surfelNorw.binding, resource: { buffer: resources.norw } },
      { binding: recycleMeta.uniforms.surfelGrid.binding, resource: { buffer: resources.grid } },
    ],
  });

  // Group 0 = uniforms (GPUVariable buffers) + G-buffer texture views (direct createView,
  // NOT GPUVariable — avoids cached-view pinning across resize). Rebuilt on recreate().
  function buildSpawnGroup0(): GPUBindGroup {
    return device.createBindGroup({
      layout: spawnPipeline.getBindGroupLayout(0),
      entries: [
        spawnShader.uniforms.invViewProj.getBindGroupEntry(device),
        spawnShader.uniforms.packA.getBindGroupEntry(device),
        spawnShader.uniforms.packB.getBindGroupEntry(device),
        spawnShader.uniforms.packC.getBindGroupEntry(device),
        { binding: depthBinding, resource: depthTex.createView() },
        { binding: normalBinding, resource: normalTex.createView() },
      ],
    });
  }
  let spawnGroup0 = buildSpawnGroup0();

  // ===== DEBUG-DRAW render pipeline (explicit layout: groups {0, 1} contiguous) =====
  const debugShader = new GPUShader(debugMeta);
  const debugPipeline = debugShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targetFormat: "bgra8unorm", // matches worldLitTexture AND frame.renderTexture
    withBlending: false,
  });
  // Group 0 = uniforms (uViewProj, uParams) — through GPUVariable (getBindGroup builds it).
  const debugGroup0 = debugShader.getBindGroup(device, 0);
  // Group 1 = the standalone surfel buffers (StorageRead). Manual entries against the
  // reflected layout (the explicit layout packs {0,1} contiguously, so this is valid).
  const debugGroup1 = device.createBindGroup({
    layout: debugPipeline.getBindGroupLayout(1),
    entries: [
      { binding: debugMeta.uniforms.posr.binding, resource: { buffer: resources.posr } },
      { binding: debugMeta.uniforms.norw.binding, resource: { buffer: resources.norw } },
    ],
  });

  // ===== uniform uploads =====
  // packA = (resW, resH, frameIndex, spawnChance);
  // packB = (surfelRadius, cap, cellSize, coverageThreshold);
  // packC = (gridCap, cellK, 0, 0).
  function uploadSpawnUniforms(frameIndex: number) {
    mat4.invert(invViewProj, viewProjMatrix);
    const m = getTypeTypedArray(spawnShader.uniforms.invViewProj.variable.type);
    m.set(invViewProj as Float32Array);
    device.queue.writeBuffer(spawnShader.uniforms.invViewProj.getGPUBuffer(device), 0, m);

    const { cellSize, coverageThreshold } = deriveCoverage(p.surfelRadius, p.coverage);
    writeVec4Raw(spawnShader, "packA", depthTex.width, depthTex.height, frameIndex, p.spawnChance);
    writeVec4Raw(spawnShader, "packB", p.surfelRadius, resources.cap, cellSize, coverageThreshold);
    writeVec4Raw(spawnShader, "packC", GRID_CAP, CELL_K, 0, 0);
  }

  // insert params = (cellSize, cap, gridCap, cellK). Re-uploaded each insert so GUI
  // edits to radius/coverage take effect live (cellSize is derived from both).
  function uploadInsertUniforms() {
    const { cellSize } = deriveCoverage(p.surfelRadius, p.coverage);
    writeVec4Raw(insertShader, "params", cellSize, resources.cap, GRID_CAP, CELL_K);
  }

  // recycle params = (cellSize, gridCap, cellK, frameIndex);
  // recycle params2 = (cap, markerDecay, recycleCoverage, 0).
  function uploadRecycleUniforms(frameIndex: number) {
    const { cellSize, recycleCoverage } = deriveCoverage(p.surfelRadius, p.coverage);
    writeVec4Raw(recycleShader, "params", cellSize, GRID_CAP, CELL_K, frameIndex);
    writeVec4Raw(recycleShader, "params2", resources.cap, p.markerDecay, recycleCoverage, 0);
  }

  // uParams = (resW, resH, quadPx, 0). uViewProj uploaded each frame (camera moves).
  function uploadDebugUniforms() {
    const m = getTypeTypedArray(debugShader.uniforms.viewProj.variable.type);
    m.set(viewProjMatrix as Float32Array);
    device.queue.writeBuffer(debugShader.uniforms.viewProj.getGPUBuffer(device), 0, m);

    writeVec4Raw(debugShader, "params", depthTex.width, depthTex.height, p.quadPx, 0);
  }

  function writeVec4Raw(
    shader: GPUShader<any>,
    key: string,
    x: number,
    y: number,
    z: number,
    w: number,
  ) {
    const buffer = getTypeTypedArray(shader.uniforms[key].variable.type);
    buffer[0] = x;
    buffer[1] = y;
    buffer[2] = z;
    buffer[3] = w;
    device.queue.writeBuffer(shader.uniforms[key].getGPUBuffer(device), 0, buffer);
  }

  // ===== public API =====

  // Zero the hash grid for this frame (encoder-ordered, before insert).
  function clearGrid(encoder: GPUCommandEncoder) {
    resources.clearGrid(encoder);
  }

  // INSERT: one thread per surfel slot; rebuild the grid from the current live set.
  function insert(encoder: GPUCommandEncoder) {
    uploadInsertUniforms();
    const pass = encoder.beginComputePass();
    pass.setPipeline(insertPipeline);
    pass.setBindGroup(0, insertGroup0);
    pass.setBindGroup(2, insertGroup2);
    pass.dispatchWorkgroups(Math.ceil(SURFEL_CAP / 64));
    pass.end();
  }

  function spawn(encoder: GPUCommandEncoder, frameIndex: number) {
    uploadSpawnUniforms(frameIndex);
    const pass = encoder.beginComputePass();
    pass.setPipeline(spawnPipeline);
    pass.setBindGroup(0, spawnGroup0);
    pass.setBindGroup(2, spawnGroup2);
    // One thread per G-buffer pixel; workgroup 8x8.
    pass.dispatchWorkgroups(Math.ceil(depthTex.width / 8), Math.ceil(depthTex.height / 8));
    pass.end();
  }

  // RECYCLE: one thread per surfel slot; age markers + push dead ids back to the stack.
  function recycle(encoder: GPUCommandEncoder, frameIndex: number) {
    uploadRecycleUniforms(frameIndex);
    const pass = encoder.beginComputePass();
    pass.setPipeline(recyclePipeline);
    pass.setBindGroup(0, recycleGroup0);
    pass.setBindGroup(2, recycleGroup2);
    pass.dispatchWorkgroups(Math.ceil(SURFEL_CAP / 64));
    pass.end();
  }

  // Full per-frame surfel update in command order: clearGrid -> insert -> spawn ->
  // recycle. Stack safety: spawn POPS (atomicAdd) and recycle PUSHES (atomicSub) in
  // SEPARATE encoder passes, so they never interleave on stack[0] within one pass.
  function run(encoder: GPUCommandEncoder, frameIndex: number) {
    clearGrid(encoder);
    insert(encoder);
    spawn(encoder, frameIndex);
    recycle(encoder, frameIndex);
  }

  // Draw into the CURRENT (caller-opened) render pass — overlay onto the presented texture.
  function drawDebug(pass: GPURenderPassEncoder) {
    uploadDebugUniforms();
    pass.setPipeline(debugPipeline);
    pass.setBindGroup(0, debugGroup0);
    pass.setBindGroup(1, debugGroup1);
    pass.draw(6, SURFEL_CAP, 0, 0);
  }

  // Swap the G-buffer textures after a canvas resize (the buffers persist).
  function recreate(nextDepth: GPUTexture, nextNormal: GPUTexture) {
    depthTex = nextDepth;
    normalTex = nextNormal;
    spawnGroup0 = buildSpawnGroup0();
  }

  // Reset the free-id stack + zero posr (all slots dead) so the debug draw clears.
  // Also zero the hash grid (the next frame's insert rebuilds it, but a one-off
  // encoder keeps clear() self-contained if it runs outside the frame loop).
  function clear() {
    resources.clear();
    const encoder = device.createCommandEncoder();
    resources.clearGrid(encoder);
    device.queue.submit([encoder.finish()]);
  }

  // Read back stack[0] = current allocated count (saturates at CAP; no recycle yet).
  async function readCount(): Promise<number> {
    const staging = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(resources.stack, 0, staging, 0, 4);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const count = new Uint32Array(staging.getMappedRange().slice(0))[0];
    staging.unmap();
    staging.destroy();
    return Math.min(count, SURFEL_CAP);
  }

  function setParams(partial: Partial<SurfelParams>) {
    Object.assign(p, partial);
    // Uniforms are re-uploaded per frame in spawn()/drawDebug(), so nothing to push here.
  }

  function destroy() {
    resources.destroy();
    spawnShader.destroy();
    debugShader.destroy();
    insertShader.destroy();
    recycleShader.destroy();
  }

  return {
    run,
    clearGrid,
    insert,
    spawn,
    recycle,
    drawDebug,
    recreate,
    clear,
    readCount,
    setParams,
    destroy,
    params: p,
  };
}
