import { mat4 } from "gl-matrix";
import { GPUShader } from "../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../Shader/index.ts";
import { viewProjMatrix } from "../ResizeSystem.ts";
import { shaderMeta as spawnMeta } from "./surfelSpawn.shader.ts";
import { shaderMeta as debugMeta } from "./surfelDebug.shader.ts";
import { createSurfelResources, SURFEL_CAP } from "./surfelResources.ts";

// Surfel Radiance Cascades — Stage A SYSTEM.
//
// Owns the three standalone surfel buffers (surfelResources) + two GPU passes:
//   spawn(encoder, frameIndex) — a COMPUTE pass over the G-buffer that, gated by a
//     low per-pixel random chance, allocates surfels onto visible surfaces (world
//     pos reconstructed from reverse-Z depth + inverse VP). autoLayout pipeline:
//     groups are {0 (uniforms+textures), 2 (StorageWrite surfel buffers)} — NON
//     contiguous, so autoLayout (WGSL @group reflection) is mandatory; the explicit
//     pipeline-layout path would mis-pack {0,2} -> {0,1}.
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
  surfelRadius: number; // FIXED world-unit radius (ortho => no depth scaling)
  quadPx: number; // debug dot diameter in screen pixels (zoom-independent)
};

export const DEFAULT_SURFEL_PARAMS: SurfelParams = {
  spawnChance: 0.02,
  surfelRadius: 0.5,
  quadPx: 6,
};

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

  // Group 2 = the three standalone surfel buffers (bindings 0/1/2, fixed in the meta).
  // Built ONCE — the buffers never change identity across resize.
  const spawnGroup2 = device.createBindGroup({
    layout: spawnPipeline.getBindGroupLayout(2),
    entries: [
      { binding: spawnMeta.uniforms.surfelStack.binding, resource: { buffer: resources.stack } },
      { binding: spawnMeta.uniforms.surfelPosr.binding, resource: { buffer: resources.posr } },
      { binding: spawnMeta.uniforms.surfelNorw.binding, resource: { buffer: resources.norw } },
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
  // packA = (resW, resH, frameIndex, spawnChance); packB = (surfelRadius, cap, 0, 0).
  function uploadSpawnUniforms(frameIndex: number) {
    mat4.invert(invViewProj, viewProjMatrix);
    const m = getTypeTypedArray(spawnShader.uniforms.invViewProj.variable.type);
    m.set(invViewProj as Float32Array);
    device.queue.writeBuffer(spawnShader.uniforms.invViewProj.getGPUBuffer(device), 0, m);

    writeVec4Raw(spawnShader, "packA", depthTex.width, depthTex.height, frameIndex, p.spawnChance);
    writeVec4Raw(spawnShader, "packB", p.surfelRadius, resources.cap, 0, 0);
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
  function clear() {
    resources.clear();
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
  }

  return {
    spawn,
    drawDebug,
    recreate,
    clear,
    readCount,
    setParams,
    destroy,
    params: p,
  };
}
