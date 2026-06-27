import { mat4, vec3 } from "gl-matrix";
import { GPUShader } from "../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../Shader/index.ts";
import { viewProjMatrix } from "../ResizeSystem.ts";
import { shaderMeta as voxelizeMeta, WORKGROUP, WORKGROUP_1D } from "./voxelize.shader.ts";
import { shaderMeta as coneMeta } from "./voxelCone.shader.ts";
import { shaderMeta as compositeMeta } from "./voxelComposite.shader.ts";
import { shaderMeta as mipMeta, WORKGROUP as MIP_WG } from "./voxelMip.shader.ts";
import { shaderMeta as probeMeta, WORKGROUP as PROBE_WG } from "./voxelProbe.shader.ts";
import { shaderMeta as sunShadowMeta } from "./sunShadow.shader.ts";
import {
  createProbeTextures,
  createVoxelTextures,
  DEFAULT_PROBE_DIMS,
  DEFAULT_VOXEL_GRID,
  voxelMipLevelCount,
  type ProbeTextures,
  type VoxelGridConfig,
  type VoxelTextures,
} from "./voxelResources.ts";
import type { SceneInstances } from "../SDFSystem/createDrawShapeSystem.ts";
import { SunLight } from "../SunLight.ts";

export type VoxelConeParams = {
  normalBias: number; // extra lift of the cone origin off the surface
  maxDist: number; // cone reach (world units)
  aperture: number; // tan(halfAngle) — cone half-angle (~0.577 = 60° full angle)
  giStrength: number; // multiplier on the probe bounce (indirect) term
  emitterDirect: number; // multiplier on the summed emitter aimed-cone DIRECT light (vs the sun)
  emitterFalloff: number; // emitter distance falloff coefficient (0 = none/flat, 1 = standard 1/d²)
};

export type VoxelCompositeParams = {
  ambient: number; // ambient floor (scaled by the cone's AO term)
  exposure: number; // HDR exposure multiplier applied before the ACES tonemap
  penumbra: number; // sun shadow softening strength: PCF filter widens as sun intensity drops below 1
};

export type VoxelProbeParams = {
  conesPerProbe: number; // full-sphere cones per probe; SH-L1 saturates ~16, so more only cuts noise
  aoConeCount: number; // short per-pixel hemisphere occlusion cones for contact AO (0 = no AO)
  aoReach: number; // AO cone reach (world units) — short, near-field contact occlusion
  aoSteps: number; // AO cone march budget (short)
};

// Voxel scene system: voxelize() fills the 3D albedo/emission/radiance textures from the SDF
// scene each frame; mips() builds the radiance pyramid; cone() gathers indirect light (N-cone
// VCT); sunDepth() renders the sun-POV shadow map; composite() produces the final lit image
// (see docs/voxel-cone-tracing-impl.md).
//
// GRANULARITY: the world box (origin + extent) is FIXED; cellSize controls voxel size
// (and thus per-axis dims = round(extent/cellSize)) — the "graininess" knob. Smaller
// cellSize = finer voxels = more of them. setCellSize() rebuilds the textures + the two
// texture-referencing bind groups; the canvas-sized output texture is independent of it.
export function createVoxelSystem({
  device,
  canvas,
  sceneInstances,
  depthTexture,
  normalTexture,
  albedoTexture,
  emissionTexture,
  grid = DEFAULT_VOXEL_GRID,
}: {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  sceneInstances: SceneInstances;
  // G-buffer (the SDF draw pass output): reverse-Z depth + world normal + albedo + per-pixel
  // self-emission. The cone pass reads depth + normal to reconstruct P + N; the composite
  // reads albedo + emission.
  depthTexture: GPUTexture;
  normalTexture: GPUTexture;
  albedoTexture: GPUTexture;
  emissionTexture: GPUTexture;
  grid?: VoxelGridConfig;
}) {
  const coneParams: VoxelConeParams = {
    normalBias: 0,
    maxDist: 24,
    aperture: 0.577,
    giStrength: 1,
    emitterDirect: 1,
    emitterFalloff: 1,
  };
  // giStrength stays in coneParams (baked into the cone's rgb); the composite only adds the
  // ambient floor.
  const compositeParams: VoxelCompositeParams = { ambient: 0.05, exposure: 1, penumbra: 4 };
  // Probe volume (fill/bounce) + per-pixel contact AO. The bounce is stored as SH-L1 (only 4
  // coeffs/channel), which ~16 cones already fully determine — so conesPerProbe past ~16-32 adds
  // nothing visible, only probe-pass cost. 32 is a safe, cheap default. AO is a few short cones.
  const probeParams: VoxelProbeParams = {
    conesPerProbe: 32,
    aoConeCount: 4,
    aoReach: 6,
    aoSteps: 12,
  };

  // LIGHTING MODEL:
  //  - emitters (point lights) → injected into the voxel volume → gathered by the cone GI
  //    (aimed + fill cones). This is the composite's 'indirect' term.
  //  - directional sun (SunLight) → a DIRECT term in the composite (N·L) with a crisp cast shadow
  //    from the sun-POV depth map (sunDepth pass). It is also injected (shadowed) into the volume
  //    by voxelize, so it contributes a GI bounce too. Dormant when SunLight is disabled (sun.w==0).

  // G-buffer textures (reassigned by recreate() on canvas resize).
  let gDepth = depthTexture;
  let gNormal = normalTexture;
  let gAlbedo = albedoTexture;
  let gEmission = emissionTexture;

  // Fixed world box (min corner + extent), derived from the initial config. cellSize is
  // the only thing that varies; dims follow from it.
  const originX = grid.originX;
  const originY = grid.originY;
  const originZ = grid.originZ;
  const extentX = grid.dimX * grid.cellSize;
  const extentY = grid.dimY * grid.cellSize;
  const extentZ = grid.dimZ * grid.cellSize;

  // ===== Shaders + pipelines (created once; only textures/bind groups rebuild). =====
  const voxShader = new GPUShader(voxelizeMeta);
  // Two compute pipelines from the one shader: `clear` zeroes the full volume (one thread per
  // voxel), `main` is the per-shape scatter (one thread per (instance, voxel-in-AABB) pair).
  // They share one pipeline layout (groups 0/1/2 — clear binds the same groups, harmless since
  // it does not read the aabb* / scene buffers).
  const voxClearPipeline = voxShader.getComputePipeline(device, "clear");
  const voxPipeline = voxShader.getComputePipeline(device, "main");
  // VCT cone GI — N-cone diffuse hemisphere gather over the G-buffer → HALF-res HDR target
  // (composite bilinear-upsamples to full res; the heavy cone work runs at ¼ the pixels).
  const coneShader = new GPUShader(coneMeta);
  const conePipeline = coneShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targetFormat: "rgba16float",
    withBlending: false,
  });
  // VCT composite (Layer 4): final = albedo·(ambient·AO + directSun + indirect) + emission.
  const compositeShader = new GPUShader(compositeMeta);
  const compositePipeline = compositeShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targetFormat: "rgba16float",
    withBlending: false,
  });
  // Voxel-radiance mip-pyramid downsample (one dispatch per level).
  const mipShader = new GPUShader(mipMeta);
  const mipPipeline = mipShader.getComputePipeline(device, "main");
  // The mip shader uses groups 0 (uniform+src) and 2 (dst storage) with NOTHING in group 1,
  // so its pipeline layout has an empty layout at index 1. Bind a matching empty group there
  // each dispatch, so strict implementations that require every layout index to be set are
  // satisfied. (The layout object is the same one the pipeline layout reflects.)
  const mipEmptyGroup1 = device.createBindGroup({
    layout: mipShader.createBindGroupLayout(device, 1),
    entries: [],
  });

  // Irradiance-probe pass: one thread per probe traces full-sphere cones into voxelRadiance and
  // writes SH-L1 (3 textures). group0 = uniforms + voxelRadiance + sampler, group2 = the SH
  // outputs, group1 empty (same situation as the mip pass → bind a matching empty group).
  const probeShader = new GPUShader(probeMeta);
  const probePipeline = probeShader.getComputePipeline(device, "main");
  const probeEmptyGroup1 = device.createBindGroup({
    layout: probeShader.createBindGroupLayout(device, 1),
    entries: [],
  });

  // Filtering sampler for textureSampleLevel over the rgba16float voxelRadiance pyramid AND the
  // trilinear probe-SH fetch.
  const voxelSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    mipmapFilter: "linear",
  });

  // Probe SH-L1 volume (3 channel textures). Resolution is independent of cellSize → created ONCE
  // here; the bind groups that reference voxelRadiance (which IS rebuilt on cellSize change) are
  // rebuilt in buildGrid via buildProbeGroups().
  const probeTextures: ProbeTextures = createProbeTextures(device, DEFAULT_PROBE_DIMS);
  const probeDimsVal = DEFAULT_PROBE_DIMS;

  // ===== Sun shadow map (depth-only pass from the sun's POV; grid/camera-independent). =====
  // Standard depth (orthoZO [0,1]) so the composite's shadow test is the simple "fragment
  // depth > stored ⇒ shadowed". Pipeline uses depthCompare "less-equal" + clear 1.0, fully
  // decoupled from the main camera's reverse-Z. Built ONCE; only sunViewProj/rayDir refresh.
  // 2048² over the 64-unit world box ≈ 0.03 world/texel — crisp enough for small objects, and
  // 4× cheaper to render + store than 4096² (depth32float: 4096²=67 MB → 2048²=17 MB). The map
  // re-renders every frame because the scene is dynamic (its content depends on object positions,
  // not just the sun direction — so it cannot be cached across frames while objects move).
  const SHADOW_SIZE = 2048;
  const sunShadowShader = new GPUShader(sunShadowMeta);
  const sunShadowPipeline = sunShadowShader.getRenderPipeline(device, "vs_main", "fs_depth", {
    withDepth: true,
    depthCompare: "less-equal",
    targets: [], // depth-only: no color attachments
  });
  const sunDepthTexture = device.createTexture({
    size: [SHADOW_SIZE, SHADOW_SIZE, 1],
    format: "depth32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  // Cached once (the texture never changes): used both as the sunDepth() depth attachment and
  // as the composite shadow-map binding — a per-frame createView() would just feed the GC.
  const sunDepthView = sunDepthTexture.createView();

  // Sun bind groups (scene buffers are stable → build ONCE, mirroring voxGroup0/voxGroup1).
  // group0 = sun viewProj + rayDir uniforms; group1 = the 7 scene-instance buffers bound by
  // BINDING NUMBER (the StorageRead declaration order matches sceneInstances.* exactly).
  const sunGroup0 = device.createBindGroup({
    layout: sunShadowPipeline.getBindGroupLayout(0),
    entries: [
      sunShadowShader.uniforms.viewProj.getBindGroupEntry(device),
      sunShadowShader.uniforms.rayDir.getBindGroupEntry(device),
    ],
  });
  const sunGroup1 = device.createBindGroup({
    layout: sunShadowPipeline.getBindGroupLayout(1),
    entries: [
      {
        binding: sunShadowMeta.uniforms.transform.binding,
        resource: { buffer: sceneInstances.transform.getGPUBuffer(device) },
      },
      {
        binding: sunShadowMeta.uniforms.kind.binding,
        resource: { buffer: sceneInstances.kind.getGPUBuffer(device) },
      },
      {
        binding: sunShadowMeta.uniforms.values.binding,
        resource: { buffer: sceneInstances.values.getGPUBuffer(device) },
      },
      {
        binding: sunShadowMeta.uniforms.roundness.binding,
        resource: { buffer: sceneInstances.roundness.getGPUBuffer(device) },
      },
      {
        binding: sunShadowMeta.uniforms.heights.binding,
        resource: { buffer: sceneInstances.heights.getGPUBuffer(device) },
      },
      {
        binding: sunShadowMeta.uniforms.color.binding,
        resource: { buffer: sceneInstances.color.getGPUBuffer(device) },
      },
      {
        binding: sunShadowMeta.uniforms.material.binding,
        resource: { buffer: sceneInstances.material.getGPUBuffer(device) },
      },
    ],
  });

  // Group 0 (voxelize) = grid uniforms; Group 1 = the 7 scene-instance buffers. Both
  // reference stable buffers (uniform + draw system's GPUVariables) → built ONCE. Scene
  // buffers are bound at the VOXELIZE meta's binding numbers (NOT
  // sceneInstances.X.getBindGroupEntry(), which carries the DRAW shader's bindings).
  const voxGroup0 = device.createBindGroup({
    layout: voxPipeline.getBindGroupLayout(0),
    entries: [
      voxShader.uniforms.gridOrigin.getBindGroupEntry(device),
      voxShader.uniforms.gridDims.getBindGroupEntry(device),
      voxShader.uniforms.instanceCount.getBindGroupEntry(device),
      voxShader.uniforms.sun.getBindGroupEntry(device),
      voxShader.uniforms.sunColor.getBindGroupEntry(device),
      voxShader.uniforms.sunViewProj.getBindGroupEntry(device),
      voxShader.uniforms.dispatch.getBindGroupEntry(device),
      // Sun shadow map: the sun-POV depth texture, sampled to shadow the injected directional sun.
      { binding: voxelizeMeta.uniforms.shadowMap.binding, resource: sunDepthView },
    ],
  });
  const voxGroup1 = device.createBindGroup({
    layout: voxPipeline.getBindGroupLayout(1),
    entries: [
      {
        binding: voxelizeMeta.uniforms.transform.binding,
        resource: { buffer: sceneInstances.transform.getGPUBuffer(device) },
      },
      {
        binding: voxelizeMeta.uniforms.kind.binding,
        resource: { buffer: sceneInstances.kind.getGPUBuffer(device) },
      },
      {
        binding: voxelizeMeta.uniforms.values.binding,
        resource: { buffer: sceneInstances.values.getGPUBuffer(device) },
      },
      {
        binding: voxelizeMeta.uniforms.roundness.binding,
        resource: { buffer: sceneInstances.roundness.getGPUBuffer(device) },
      },
      {
        binding: voxelizeMeta.uniforms.heights.binding,
        resource: { buffer: sceneInstances.heights.getGPUBuffer(device) },
      },
      {
        binding: voxelizeMeta.uniforms.color.binding,
        resource: { buffer: sceneInstances.color.getGPUBuffer(device) },
      },
      {
        binding: voxelizeMeta.uniforms.material.binding,
        resource: { buffer: sceneInstances.material.getGPUBuffer(device) },
      },
      // Per-instance voxel-AABB boxes + prefix sum, built on the CPU each frame and uploaded to
      // these two new StorageRead buffers (their own GPUVariables — sized MAX_INSTANCE_COUNT*4
      // i32, STORAGE | COPY_DST). Bound at the voxelize meta's new binding numbers (7, 8).
      voxShader.uniforms.aabbMin.getBindGroupEntry(device),
      voxShader.uniforms.aabbDim.getBindGroupEntry(device),
    ],
  });

  // --- Scratch typed arrays for uniform uploads. ---
  const originArr = getTypeTypedArray(voxelizeMeta.uniforms.gridOrigin.type); // Float32Array(4)
  const dimsArr = getTypeTypedArray(voxelizeMeta.uniforms.gridDims.type); // Int32Array(4)
  const instanceCountArr = getTypeTypedArray(voxelizeMeta.uniforms.instanceCount.type); // Uint32Array(1)
  const sunArr = getTypeTypedArray(voxelizeMeta.uniforms.sun.type); // Float32Array(4)
  const sunColorArr = getTypeTypedArray(voxelizeMeta.uniforms.sunColor.type); // Float32Array(4)
  // The sun view-proj voxelize samples the shadow map with (for the shadowed sun injection).
  const voxSunViewProjArr = getTypeTypedArray(voxelizeMeta.uniforms.sunViewProj.type); // Float32Array(16)
  // Scatter dispatch: .x = total work items, .y = threads per workgroup-grid row, .z/.w spare.
  const dispatchArr = getTypeTypedArray(voxelizeMeta.uniforms.dispatch.type); // Int32Array(4)
  // Per-instance AABB scratch (allocated ONCE — never per frame). Packed vec4<i32> per instance:
  // aabbMinArr[k*4 + 0..2] = voxel box min, +3 = prefix start; aabbDimArr[k*4 + 0..2] = dims, +3 = n.
  const aabbMinArr = getTypeTypedArray(voxelizeMeta.uniforms.aabbMin.type) as Int32Array; // Int32Array(MAX*4)
  const aabbDimArr = getTypeTypedArray(voxelizeMeta.uniforms.aabbDim.type) as Int32Array; // Int32Array(MAX*4)
  const invViewProj = mat4.create(); // reused by cone() + composite() for inverse-viewProj
  // Cone scratch.
  const coneParamsArr = getTypeTypedArray(coneMeta.uniforms.params.type); // Float32Array(4)
  const coneParams2Arr = getTypeTypedArray(coneMeta.uniforms.params2.type); // Float32Array(4)
  const coneAoArr = getTypeTypedArray(coneMeta.uniforms.aoParams.type); // Float32Array(4)
  const coneInvArr = getTypeTypedArray(coneMeta.uniforms.invViewProj.type); // Float32Array(16)
  // Probe scratch.
  const probeOriginArr = getTypeTypedArray(probeMeta.uniforms.gridOrigin.type); // Float32Array(4)
  const probeGridDimsArr = getTypeTypedArray(probeMeta.uniforms.gridDims.type); // Int32Array(4)
  const probeDimsArr = getTypeTypedArray(probeMeta.uniforms.probeDims.type); // Int32Array(4)
  const probeParamsArr = getTypeTypedArray(probeMeta.uniforms.probeParams.type); // Float32Array(4)
  // Auto-discovered emitter centers the cone importance-samples (x,y,z,radius per light) +
  // parallel colors (r,g,b,intensity) for the analytic-direct shadow term.
  const coneLightsArr = getTypeTypedArray(coneMeta.uniforms.lights.type); // Float32Array(32)
  const coneLightColorsArr = getTypeTypedArray(coneMeta.uniforms.lightColor.type); // Float32Array(32)
  let coneLightCount = 0;
  // Composite scratch.
  const compParamsArr = getTypeTypedArray(compositeMeta.uniforms.params.type); // Float32Array(4)
  const compParams2Arr = getTypeTypedArray(compositeMeta.uniforms.params2.type); // Float32Array(4)
  const compSunArr = getTypeTypedArray(compositeMeta.uniforms.sun.type); // Float32Array(4)
  const compSunColorArr = getTypeTypedArray(compositeMeta.uniforms.sunColor.type); // Float32Array(4)
  const compInvArr = getTypeTypedArray(compositeMeta.uniforms.invViewProj.type); // Float32Array(16)
  const compSunViewProjArr = getTypeTypedArray(compositeMeta.uniforms.sunViewProj.type); // Float32Array(16)
  // World units per shadow texel (sun ortho width / SHADOW_SIZE) → composite normal-offset bias.
  let sunWorldTexel = 0;
  // Mip scratch: .xyz = destination mip dims (re-uploaded per level).
  const mipArr = getTypeTypedArray(mipMeta.uniforms.mip.type); // Int32Array(4)

  // Sun shadow scratch (allocate ONCE — never per frame). sunViewProj is computed each frame
  // from SunLight + the grid AABB and uploaded to BOTH the sunShadow shader (vs uViewProj)
  // and the composite (uSunViewProj). rayDir = sun travel direction (= -dirTowardSun).
  const sunView = mat4.create();
  const sunProj = mat4.create();
  const sunViewProj = mat4.create();
  const sunEye = vec3.create();
  const sunCenter = vec3.create();
  const sunUp = vec3.create();
  const sunCorner = vec3.create();
  const sunViewProjArr = getTypeTypedArray(sunShadowMeta.uniforms.viewProj.type); // Float32Array(16)
  const sunRayDirArr = getTypeTypedArray(sunShadowMeta.uniforms.rayDir.type); // Float32Array(4)

  // --- Grid state (rebuilt by buildGrid). ---
  let cellSize = grid.cellSize;
  let dimX = grid.dimX;
  let dimY = grid.dimY;
  let dimZ = grid.dimZ;
  let textures: VoxelTextures;
  let voxGroup2: GPUBindGroup;
  let coneGroup0: GPUBindGroup;
  let compositeGroup0: GPUBindGroup;
  // Probe bind groups: group0 (uniforms + voxelRadiance all-mips view + sampler) is rebuilt with
  // the voxelRadiance texture in buildGrid; group2 (SH outputs) references the persistent probe
  // textures and is built once but is convenient to rebuild alongside.
  let probeGroup0: GPUBindGroup;
  let probeGroup2: GPUBindGroup;
  let dispatchX = 0;
  let dispatchY = 0;
  let dispatchZ = 0;
  // Scatter dispatch (rebuilt every frame from the prefix-sum total in voxelize()).
  let scatterTotal = 0;
  let scatterDispatchX = 0;
  let scatterDispatchY = 0;

  // Mip-pyramid state (rebuilt by buildGrid). One downsample step per pair of adjacent
  // levels → mipCount-1 steps, indexed 0..mipCount-2. mipGroup0[L]/mipGroup2[L] downsample
  // mip L → mip L+1.
  let mipCount = 1;
  let mipBuf: GPUBuffer[] = [];
  let mipGroup0: GPUBindGroup[] = [];
  let mipGroup2: GPUBindGroup[] = [];

  // (Re)build the Layer-2 cone bind group: uniforms + the G-buffer (depth/normal) + the
  // ALL-mips voxelRadiance view + the shared filtering sampler. Rebuilt whenever the
  // voxelRadiance view changes (grid rebuild) or the G-buffer changes (canvas resize).
  function buildConeGroup() {
    coneGroup0 = device.createBindGroup({
      layout: conePipeline.getBindGroupLayout(0),
      entries: [
        coneShader.uniforms.params.getBindGroupEntry(device),
        coneShader.uniforms.params2.getBindGroupEntry(device),
        coneShader.uniforms.aoParams.getBindGroupEntry(device),
        coneShader.uniforms.invViewProj.getBindGroupEntry(device),
        coneShader.uniforms.gridOrigin.getBindGroupEntry(device),
        coneShader.uniforms.gridDims.getBindGroupEntry(device),
        coneShader.uniforms.lights.getBindGroupEntry(device),
        coneShader.uniforms.lightColor.getBindGroupEntry(device),
        { binding: coneMeta.uniforms.depthTex.binding, resource: gDepth.createView() },
        { binding: coneMeta.uniforms.normalTex.binding, resource: gNormal.createView() },
        // ALL-mips sampled view so textureSampleLevel can pick any lod.
        {
          binding: coneMeta.uniforms.voxelRadiance.binding,
          resource: textures.voxelRadiance.createView({ dimension: "3d" }),
        },
        // Irradiance-probe SH-L1 volume (one sampled 3D view per channel) for the fill term.
        {
          binding: coneMeta.uniforms.shR.binding,
          resource: probeTextures.shR.createView({ dimension: "3d" }),
        },
        {
          binding: coneMeta.uniforms.shG.binding,
          resource: probeTextures.shG.createView({ dimension: "3d" }),
        },
        {
          binding: coneMeta.uniforms.shB.binding,
          resource: probeTextures.shB.createView({ dimension: "3d" }),
        },
        { binding: coneMeta.uniforms.voxelSampler.binding, resource: voxelSampler },
      ],
    });
  }

  // (Re)build the probe bind groups: group0 = uniforms + the ALL-mips voxelRadiance view + the
  // sampler; group2 = the three SH storage views (write-only, single mip). group0 references
  // voxelRadiance, so it must be rebuilt whenever the voxelRadiance texture is recreated (grid).
  function buildProbeGroups() {
    probeGroup0 = device.createBindGroup({
      layout: probePipeline.getBindGroupLayout(0),
      entries: [
        probeShader.uniforms.gridOrigin.getBindGroupEntry(device),
        probeShader.uniforms.gridDims.getBindGroupEntry(device),
        probeShader.uniforms.probeDims.getBindGroupEntry(device),
        probeShader.uniforms.probeParams.getBindGroupEntry(device),
        {
          binding: probeMeta.uniforms.voxelRadiance.binding,
          resource: textures.voxelRadiance.createView({ dimension: "3d" }),
        },
        { binding: probeMeta.uniforms.voxelSampler.binding, resource: voxelSampler },
      ],
    });
    probeGroup2 = device.createBindGroup({
      layout: probePipeline.getBindGroupLayout(2),
      entries: [
        {
          binding: probeMeta.uniforms.shR.binding,
          resource: probeTextures.shR.createView({ dimension: "3d" }),
        },
        {
          binding: probeMeta.uniforms.shG.binding,
          resource: probeTextures.shG.createView({ dimension: "3d" }),
        },
        {
          binding: probeMeta.uniforms.shB.binding,
          resource: probeTextures.shB.createView({ dimension: "3d" }),
        },
      ],
    });
  }

  // (Re)build the Layer-4 composite bind group: uniforms + the G-buffer (albedo/normal/
  // emission) + the cone output (indirect+AO). The coneSampler bilinear-upsamples the half-res
  // cone output. Rebuilt when the G-buffer / coneOutput change (resize or grid rebuild).
  function buildCompositeGroup() {
    compositeGroup0 = device.createBindGroup({
      layout: compositePipeline.getBindGroupLayout(0),
      entries: [
        compositeShader.uniforms.params.getBindGroupEntry(device),
        compositeShader.uniforms.params2.getBindGroupEntry(device),
        compositeShader.uniforms.sun.getBindGroupEntry(device),
        compositeShader.uniforms.sunColor.getBindGroupEntry(device),
        compositeShader.uniforms.invViewProj.getBindGroupEntry(device),
        compositeShader.uniforms.sunViewProj.getBindGroupEntry(device),
        { binding: compositeMeta.uniforms.albedoTex.binding, resource: gAlbedo.createView() },
        { binding: compositeMeta.uniforms.normalTex.binding, resource: gNormal.createView() },
        { binding: compositeMeta.uniforms.coneTex.binding, resource: coneView },
        // Linear/clamp sampler (reuse voxelSampler) for the half-res cone upsample.
        { binding: compositeMeta.uniforms.coneSampler.binding, resource: voxelSampler },
        {
          binding: compositeMeta.uniforms.emissionTex.binding,
          resource: gEmission.createView(),
        },
        // Sun shadow: reverse-Z camera depth (reconstruct P) + the sun-POV depth map.
        { binding: compositeMeta.uniforms.depthTex.binding, resource: gDepth.createView() },
        { binding: compositeMeta.uniforms.shadowMap.binding, resource: sunDepthView },
      ],
    });
  }

  // (Re)build the voxel textures + the two texture-referencing bind groups for the
  // current cellSize, and upload the grid uniforms to all shaders.
  function buildGrid(newCellSize: number) {
    cellSize = newCellSize;
    dimX = Math.max(1, Math.round(extentX / cellSize));
    dimY = Math.max(1, Math.round(extentY / cellSize));
    dimZ = Math.max(1, Math.round(extentZ / cellSize));

    textures = createVoxelTextures(device, {
      originX,
      originY,
      originZ,
      dimX,
      dimY,
      dimZ,
      cellSize,
    });

    // Group 2 (voxelize) = voxel output storage textures (write-only, dimension 3d).
    // voxelRadiance now has a mip pyramid; a storage view MUST span exactly one mip → bind
    // mip 0 only (the voxelize pass writes level 0; voxelMip builds the rest).
    voxGroup2 = device.createBindGroup({
      layout: voxPipeline.getBindGroupLayout(2),
      entries: [
        {
          binding: voxelizeMeta.uniforms.voxelRadiance.binding,
          resource: textures.voxelRadiance.createView({
            dimension: "3d",
            baseMipLevel: 0,
            mipLevelCount: 1,
          }),
        },
      ],
    });

    // Mip-pyramid downsample groups (mip L → L+1). srcView is a single-mip SAMPLED view of
    // mip L; dstView is a single-mip STORAGE view of mip L+1 (different subresources of the
    // same texture → allowed). Rebuilt here because views/buffers depend on the new dims.
    for (const b of mipBuf) b.destroy();
    mipCount = voxelMipLevelCount(dimX, dimY, dimZ);
    mipBuf = [];
    mipGroup0 = [];
    mipGroup2 = [];
    for (let L = 0; L < mipCount - 1; L++) {
      const buf = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      mipBuf.push(buf);

      const srcView = textures.voxelRadiance.createView({
        dimension: "3d",
        baseMipLevel: L,
        mipLevelCount: 1,
      });
      const dstView = textures.voxelRadiance.createView({
        dimension: "3d",
        baseMipLevel: L + 1,
        mipLevelCount: 1,
      });

      mipGroup0.push(
        device.createBindGroup({
          layout: mipPipeline.getBindGroupLayout(0),
          entries: [
            { binding: mipMeta.uniforms.mip.binding, resource: { buffer: buf } },
            { binding: mipMeta.uniforms.src.binding, resource: srcView },
          ],
        }),
      );
      mipGroup2.push(
        device.createBindGroup({
          layout: mipPipeline.getBindGroupLayout(2),
          entries: [{ binding: mipMeta.uniforms.dst.binding, resource: dstView }],
        }),
      );

      // Destination dims = mip L+1 dims (halved per axis, floored at 1).
      mipArr[0] = Math.max(1, dimX >> (L + 1));
      mipArr[1] = Math.max(1, dimY >> (L + 1));
      mipArr[2] = Math.max(1, dimZ >> (L + 1));
      mipArr[3] = 0;
      device.queue.writeBuffer(buf, 0, mipArr);
    }

    // Cone bind group references the rebuilt voxelRadiance view + the (stable) G-buffer.
    buildConeGroup();
    // Composite bind group references the G-buffer (albedo/normal/emission/depth) + coneOutput.
    buildCompositeGroup();
    // Probe group0 references the rebuilt voxelRadiance view.
    buildProbeGroups();

    // Grid uniforms (shared by all shaders).
    originArr[0] = originX;
    originArr[1] = originY;
    originArr[2] = originZ;
    originArr[3] = cellSize;
    dimsArr[0] = dimX;
    dimsArr[1] = dimY;
    dimsArr[2] = dimZ;
    dimsArr[3] = 0;
    device.queue.writeBuffer(voxShader.uniforms.gridOrigin.getGPUBuffer(device), 0, originArr);
    device.queue.writeBuffer(voxShader.uniforms.gridDims.getGPUBuffer(device), 0, dimsArr);
    device.queue.writeBuffer(coneShader.uniforms.gridOrigin.getGPUBuffer(device), 0, originArr);
    device.queue.writeBuffer(coneShader.uniforms.gridDims.getGPUBuffer(device), 0, dimsArr);
    // (composite no longer has grid uniforms — its sun shadow uses the shadow map, not voxels.)
    // Probe shares the SAME world box (origin + cellSize + voxel dims) to map probes into it.
    probeOriginArr.set(originArr);
    probeGridDimsArr.set(dimsArr);
    probeDimsArr[0] = probeDimsVal.x;
    probeDimsArr[1] = probeDimsVal.y;
    probeDimsArr[2] = probeDimsVal.z;
    probeDimsArr[3] = 0;
    device.queue.writeBuffer(
      probeShader.uniforms.gridOrigin.getGPUBuffer(device),
      0,
      probeOriginArr,
    );
    device.queue.writeBuffer(
      probeShader.uniforms.gridDims.getGPUBuffer(device),
      0,
      probeGridDimsArr,
    );
    device.queue.writeBuffer(probeShader.uniforms.probeDims.getGPUBuffer(device), 0, probeDimsArr);

    dispatchX = Math.ceil(dimX / WORKGROUP);
    dispatchY = Math.ceil(dimY / WORKGROUP);
    dispatchZ = Math.ceil(dimZ / WORKGROUP);
  }
  // HALF-res HDR target for the Layer-2 cone gather.
  // Half-res for perf (¼ the pixels → ~4× less cone work); the composite bilinear-upsamples
  // it back to full res (indirect light is low-frequency, so that is fine). The render pass
  // viewport IS the texture size, so the cone pass renders at half res automatically.
  const createConeOutput = () =>
    device.createTexture({
      size: [
        Math.max(1, Math.ceil(canvas.width / 2)),
        Math.max(1, Math.ceil(canvas.height / 2)),
        1,
      ],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  let coneOutput = createConeOutput();
  // Cached view; used both as the cone() attachment and as the composite's sampled binding.
  // Refreshed in recreate() (the texture is rebuilt on resize, before the bind groups).
  let coneView = coneOutput.createView();

  // Full-res HDR target for the Layer-4 composite (the final lit image — "final" source).
  const createCompositeOutput = () =>
    device.createTexture({
      size: [canvas.width, canvas.height, 1],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  let compositeOutput = createCompositeOutput();
  // Cached attachment view; refreshed in recreate() when the texture is rebuilt on resize.
  let compositeView = compositeOutput.createView();

  // Built last: buildGrid() (and recreate()) call buildCompositeGroup(), which references
  // coneOutput + compositeOutput, so those textures must exist first.
  buildGrid(cellSize);

  // Compute the sun's orthographic view-projection (orthoZO, z in [0,1]) fitted to the grid
  // AABB, plus the sun travel ray. Uploads sunViewProj to BOTH the sunShadow shader and the
  // composite, and rayDir to the sunShadow shader. Recomputed each frame (sun/GUI can move).
  function buildSunViewProj() {
    const minX = originX;
    const minY = originY;
    const minZ = originZ;
    const maxX = originX + extentX;
    const maxY = originY + extentY;
    const maxZ = originZ + extentZ;
    const cx = (minX + maxX) * 0.5;
    const cy = (minY + maxY) * 0.5;
    const cz = (minZ + maxZ) * 0.5;

    // Direction TOWARD the sun (unit), same formula as voxelize()/composite().
    const a = SunLight.angle;
    const e = SunLight.elevation;
    const ce = Math.cos(e);
    const sdx = Math.cos(a) * ce;
    const sdy = Math.sin(a) * ce;
    const sdz = Math.sin(e);

    // Eye pulled back from the AABB center along +dirTowardSun; the ortho near/far fit below
    // folds the distance out, so any value past the bounding-sphere radius is fine.
    const radius = 0.5 * Math.hypot(extentX, extentY, extentZ);
    const dist = radius * 2.0 + 1.0;
    sunCenter[0] = cx;
    sunCenter[1] = cy;
    sunCenter[2] = cz;
    sunEye[0] = cx + sdx * dist;
    sunEye[1] = cy + sdy * dist;
    sunEye[2] = cz + sdz * dist;

    // Up-vector degeneracy: when the sun is near-vertical (|sdz|→1) forward ∥ +Z makes lookAt
    // produce NaNs; swap up to +Y in that case.
    if (Math.abs(sdz) > 0.999) {
      sunUp[0] = 0;
      sunUp[1] = 1;
      sunUp[2] = 0;
    } else {
      sunUp[0] = 0;
      sunUp[1] = 0;
      sunUp[2] = 1;
    }
    mat4.lookAt(sunView, sunEye, sunCenter, sunUp);

    // Fit an orthographic box to the 8 AABB corners IN LIGHT/VIEW space (tight, sun-angle
    // independent). In view space the camera looks down -Z, so visible z is negative.
    let lminX = Infinity;
    let lminY = Infinity;
    let lminZ = Infinity;
    let lmaxX = -Infinity;
    let lmaxY = -Infinity;
    let lmaxZ = -Infinity;
    for (let i = 0; i < 8; i++) {
      sunCorner[0] = i & 1 ? maxX : minX;
      sunCorner[1] = i & 2 ? maxY : minY;
      sunCorner[2] = i & 4 ? maxZ : minZ;
      vec3.transformMat4(sunCorner, sunCorner, sunView);
      if (sunCorner[0] < lminX) lminX = sunCorner[0];
      if (sunCorner[0] > lmaxX) lmaxX = sunCorner[0];
      if (sunCorner[1] < lminY) lminY = sunCorner[1];
      if (sunCorner[1] > lmaxY) lmaxY = sunCorner[1];
      if (sunCorner[2] < lminZ) lminZ = sunCorner[2];
      if (sunCorner[2] > lmaxZ) lmaxZ = sunCorner[2];
    }
    // near/far are POSITIVE distances; view-space z is negative going forward, so
    // near = -lmaxZ (closest), far = -lminZ (farthest). Pad to avoid front-face clipping.
    const near = -lmaxZ - 1.0;
    const far = -lminZ + 1.0;
    mat4.orthoZO(sunProj, lminX, lmaxX, lminY, lmaxY, near, far);
    mat4.multiply(sunViewProj, sunProj, sunView);

    // World units per shadow texel (larger ortho axis / map resolution) → composite normal-offset.
    sunWorldTexel = Math.max(lmaxX - lminX, lmaxY - lminY) / SHADOW_SIZE;

    sunViewProjArr.set(sunViewProj as Float32Array);
    device.queue.writeBuffer(
      sunShadowShader.uniforms.viewProj.getGPUBuffer(device),
      0,
      sunViewProjArr,
    );
    compSunViewProjArr.set(sunViewProj as Float32Array);
    device.queue.writeBuffer(
      compositeShader.uniforms.sunViewProj.getGPUBuffer(device),
      0,
      compSunViewProjArr,
    );
    // ALSO upload to voxelize's uSunViewProj — it samples the shadow map at voxelize time and
    // MUST use the same matrix that rendered the map (sunDepth runs first in the loop).
    voxSunViewProjArr.set(sunViewProj as Float32Array);
    device.queue.writeBuffer(
      voxShader.uniforms.sunViewProj.getGPUBuffer(device),
      0,
      voxSunViewProjArr,
    );

    // Sun travel direction = -dirTowardSun (already unit). xyz dir, w unused.
    sunRayDirArr[0] = -sdx;
    sunRayDirArr[1] = -sdy;
    sunRayDirArr[2] = -sdz;
    sunRayDirArr[3] = 0;
    device.queue.writeBuffer(sunShadowShader.uniforms.rayDir.getGPUBuffer(device), 0, sunRayDirArr);
  }

  // Render the SDF scene from the sun's POV into sunDepthTexture (depth-only). MUST run after
  // prepare() (scene buffers current) and before composite() (which samples the map). Refreshes
  // the sun matrices each call, so it can run any time before composite.
  function sunDepth(encoder: GPUCommandEncoder) {
    buildSunViewProj();
    const pass = encoder.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: {
        view: sunDepthView,
        depthClearValue: 1.0, // standard depth: far = 1
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    pass.setPipeline(sunShadowPipeline);
    pass.setBindGroup(0, sunGroup0);
    pass.setBindGroup(1, sunGroup1);
    pass.draw(36, sceneInstances.instanceCount, 0, 0);
    pass.end();
  }

  // Re-voxelize the scene into the 3D textures (run before debug()/the GI gather).
  function voxelize(encoder: GPUCommandEncoder) {
    instanceCountArr[0] = sceneInstances.instanceCount;
    device.queue.writeBuffer(
      voxShader.uniforms.instanceCount.getGPUBuffer(device),
      0,
      instanceCountArr,
    );

    // Directional sun, recomputed each frame from the SunLight singleton. .xyz = world
    // dir TOWARD the sun (azimuth + elevation), .w = effective intensity (0 = disabled).
    const a = SunLight.angle;
    const e = SunLight.elevation;
    const ce = Math.cos(e);
    sunArr[0] = Math.cos(a) * ce;
    sunArr[1] = Math.sin(a) * ce;
    sunArr[2] = Math.sin(e);
    sunArr[3] = SunLight.enabled ? SunLight.intensity : 0;
    device.queue.writeBuffer(voxShader.uniforms.sun.getGPUBuffer(device), 0, sunArr);
    sunColorArr[0] = SunLight.color[0];
    sunColorArr[1] = SunLight.color[1];
    sunColorArr[2] = SunLight.color[2];
    device.queue.writeBuffer(voxShader.uniforms.sunColor.getGPUBuffer(device), 0, sunColorArr);
    // uSunViewProj is uploaded by buildSunViewProj() inside sunDepth(), which runs BEFORE
    // voxelize() in the loop (so the shadowed-sun injection samples the matching matrix).

    // ===== Build per-instance voxel AABBs + the prefix-sum work list (CPU). =====
    // The CPU arrays are filled by prepare() (which runs before voxelize() each frame), so
    // they are current. For each instance compute a CONSERVATIVE (yaw-invariant) world AABB,
    // convert it to a voxel box clamped to the grid, and accumulate the prefix sum of voxel
    // counts. The scatter shader walks this flat list via a binary search on `start`.
    const n = sceneInstances.instanceCount;
    const tr = sceneInstances.cpuTransform;
    const kindArr = sceneInstances.cpuKind;
    const valArr = sceneInstances.cpuValues;
    const roundArr = sceneInstances.cpuRoundness;
    const heightArr = sceneInstances.cpuHeights;
    let prefix = 0;
    for (let k = 0; k < n; k++) {
      // Translation is column-major mat4 elements 12,13,14 (per-instance 16-float stride).
      const cx = tr[k * 16 + 12];
      const cy = tr[k * 16 + 13];
      const tz = tr[k * 16 + 14];
      const kind = kindArr[k];
      const h = heightArr[k];
      const round = roundArr[k];

      // Conservative bounding-circle radius in XY, computed from the SAME geometry the
      // footprint uses so the AABB never clips the shape. max(|value|) is NOT a valid
      // bound for skewed/diagonal footprints, so kinds 3 (parallelogram) and 5 (triangle)
      // get the Euclidean extent of their true footprint; all others fall back to max(|value|)
      // (which covers axis-aligned half-extents and the sphere radius in values[0]).
      const v0 = valArr[k * 6 + 0];
      const v1 = valArr[k * 6 + 1];
      const v2 = valArr[k * 6 + 2];
      const v3 = valArr[k * 6 + 3];
      const v4 = valArr[k * 6 + 4];
      const v5 = valArr[k * 6 + 5];
      let rxyShape: number;
      if (kind === 3) {
        // Parallelogram: skew widens the X half-extent (halfX = width/2 + |skew|); the worst
        // corner is at hypot(halfX, height/2). values = (width, height, skew, ...).
        const halfX = v0 / 2 + Math.abs(v2);
        const halfY = v1 / 2;
        rxyShape = Math.hypot(halfX, halfY);
      } else if (kind === 5) {
        // Triangle: the 6 values are signed vertex coordinates (ax,ay,bx,by,cx,cy). The
        // conservative radius is the farthest vertex distance from the local origin.
        rxyShape = Math.max(Math.hypot(v0, v1), Math.hypot(v2, v3), Math.hypot(v4, v5));
      } else {
        let maxAbs = 0;
        for (let j = 0; j < 6; j++) {
          const v = Math.abs(valArr[k * 6 + j]);
          if (v > maxAbs) maxAbs = v;
        }
        rxyShape = maxAbs;
      }
      const rxy = rxyShape + round + cellSize;
      // Z half-extent: sphere (kind 6) uses its radius (values[0]); everything else is an
      // extruded footprint of half-height h*0.5. Expanded by roundness + one cell.
      const zHalf = (kind === 6 ? valArr[k * 6] : h * 0.5) + round + cellSize;
      const centerZ = tz + h * 0.5;

      const minX = cx - rxy;
      const maxX = cx + rxy;
      const minY = cy - rxy;
      const maxY = cy + rxy;
      const minZ = centerZ - zHalf;
      const maxZ = centerZ + zHalf;

      // World AABB -> voxel index box, clamped to [0, dim] (floor min, ceil max), then size.
      const vx0 = Math.min(Math.max(Math.floor((minX - originX) / cellSize), 0), dimX);
      const vx1 = Math.min(Math.max(Math.ceil((maxX - originX) / cellSize), 0), dimX);
      const vy0 = Math.min(Math.max(Math.floor((minY - originY) / cellSize), 0), dimY);
      const vy1 = Math.min(Math.max(Math.ceil((maxY - originY) / cellSize), 0), dimY);
      const vz0 = Math.min(Math.max(Math.floor((minZ - originZ) / cellSize), 0), dimZ);
      const vz1 = Math.min(Math.max(Math.ceil((maxZ - originZ) / cellSize), 0), dimZ);
      const nx = Math.max(0, vx1 - vx0);
      const ny = Math.max(0, vy1 - vy0);
      const nz = Math.max(0, vz1 - vz0);
      const count = nx * ny * nz;

      // Every index gets a `start` (empty ranges share their successor's start and are skipped
      // by the binary search). aabbMin.w = prefix start; aabbDim.w = voxel count.
      aabbMinArr[k * 4 + 0] = vx0;
      aabbMinArr[k * 4 + 1] = vy0;
      aabbMinArr[k * 4 + 2] = vz0;
      aabbMinArr[k * 4 + 3] = prefix;
      aabbDimArr[k * 4 + 0] = nx;
      aabbDimArr[k * 4 + 1] = ny;
      aabbDimArr[k * 4 + 2] = nz;
      aabbDimArr[k * 4 + 3] = count;
      prefix += count;
    }
    scatterTotal = prefix;

    // Upload the AABB lists (only the live n entries matter; the binary search bound is
    // uInstanceCount = n). Uploading the whole MAX-sized buffer is simplest + allocation-free.
    device.queue.writeBuffer(voxShader.uniforms.aabbMin.getGPUBuffer(device), 0, aabbMinArr);
    device.queue.writeBuffer(voxShader.uniforms.aabbDim.getGPUBuffer(device), 0, aabbDimArr);

    // Scatter dispatch sizing — 2D over workgroups to dodge the 65535 per-dim workgroup cap.
    const wgTotal = Math.ceil(scatterTotal / WORKGROUP_1D);
    scatterDispatchX = Math.min(wgTotal, 65535);
    scatterDispatchY = scatterDispatchX > 0 ? Math.ceil(wgTotal / scatterDispatchX) : 0;
    // uDispatch: .x = total work items, .y = threads per workgroup-grid row (= dispatchX * WG).
    dispatchArr[0] = scatterTotal;
    dispatchArr[1] = scatterDispatchX * WORKGROUP_1D;
    dispatchArr[2] = 0;
    dispatchArr[3] = 0;
    device.queue.writeBuffer(voxShader.uniforms.dispatch.getGPUBuffer(device), 0, dispatchArr);

    // CLEAR (full grid) then SCATTER (compacted work list), in SEPARATE compute passes so the
    // encoder barriers between them — the scatter's textureStore must see a fully-zeroed volume
    // (dispatches within ONE pass are NOT synchronized; a same-pass clear could race/clobber a
    // solid voxel). Clear ALWAYS runs so the volume is zeroed; the scatter only writes solids.
    const clearPass = encoder.beginComputePass();
    clearPass.setPipeline(voxClearPipeline);
    clearPass.setBindGroup(0, voxGroup0);
    clearPass.setBindGroup(1, voxGroup1);
    clearPass.setBindGroup(2, voxGroup2);
    clearPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    clearPass.end();

    if (scatterTotal > 0) {
      const scatterPass = encoder.beginComputePass();
      scatterPass.setPipeline(voxPipeline);
      scatterPass.setBindGroup(0, voxGroup0);
      scatterPass.setBindGroup(1, voxGroup1);
      scatterPass.setBindGroup(2, voxGroup2);
      scatterPass.dispatchWorkgroups(scatterDispatchX, scatterDispatchY, 1);
      scatterPass.end();
    }
  }

  // Build the voxelRadiance mip pyramid: one compute pass PER level (passes are ordered/
  // barriered by the encoder, so level L+1 sees level L's writes — dispatches WITHIN a pass
  // are not synchronized, hence one pass each). Must run AFTER voxelize() (level 0 reads
  // mip 0 that voxelize wrote) and in the SAME encoder.
  function mips(encoder: GPUCommandEncoder) {
    for (let L = 0; L < mipCount - 1; L++) {
      const dx = Math.max(1, dimX >> (L + 1));
      const dy = Math.max(1, dimY >> (L + 1));
      const dz = Math.max(1, dimZ >> (L + 1));
      const pass = encoder.beginComputePass();
      pass.setPipeline(mipPipeline);
      pass.setBindGroup(0, mipGroup0[L]);
      pass.setBindGroup(1, mipEmptyGroup1);
      pass.setBindGroup(2, mipGroup2[L]);
      pass.dispatchWorkgroups(
        Math.ceil(dx / MIP_WG),
        Math.ceil(dy / MIP_WG),
        Math.ceil(dz / MIP_WG),
      );
      pass.end();
    }
  }

  // Irradiance-probe pass: each probe traces conesPerProbe full-sphere cones through the
  // voxelRadiance pyramid → SH-L1 (shR/shG/shB). MUST run AFTER mips() (it samples the pyramid)
  // and BEFORE cone() (which reads the SH volume for the fill term). One compute dispatch.
  function probe(encoder: GPUCommandEncoder) {
    probeParamsArr[0] = probeParams.conesPerProbe;
    probeParamsArr[1] = coneParams.maxDist; // probe cone reach = the global GI max distance
    probeParamsArr[2] = coneParams.aperture; // same aperture as the (former) fill cones
    probeParamsArr[3] = 0;
    device.queue.writeBuffer(
      probeShader.uniforms.probeParams.getGPUBuffer(device),
      0,
      probeParamsArr,
    );

    const pass = encoder.beginComputePass();
    pass.setPipeline(probePipeline);
    pass.setBindGroup(0, probeGroup0);
    pass.setBindGroup(1, probeEmptyGroup1);
    pass.setBindGroup(2, probeGroup2);
    pass.dispatchWorkgroups(
      Math.ceil(probeDimsVal.x / PROBE_WG),
      Math.ceil(probeDimsVal.y / PROBE_WG),
      Math.ceil(probeDimsVal.z / PROBE_WG),
    );
    pass.end();
  }

  // VCT cone GI: per-pixel AIMED emitter cones (sharp direct + shadow) + a trilinear probe-SH
  // fetch for the fill/bounce + short AO cones → coneOutput (HALF-res HDR; composite bilinear-
  // upsamples). MUST run AFTER voxelize() + mips() + probe(). Reads the G-buffer (depth + normal).
  function cone(encoder: GPUCommandEncoder) {
    coneParamsArr[0] = coneParams.normalBias;
    coneParamsArr[1] = coneParams.maxDist;
    coneParamsArr[2] = coneParams.aperture;
    coneParamsArr[3] = coneParams.giStrength;
    device.queue.writeBuffer(coneShader.uniforms.params.getGPUBuffer(device), 0, coneParamsArr);

    coneParams2Arr[0] = canvas.width;
    coneParams2Arr[1] = canvas.height;
    coneParams2Arr[2] = coneParams.emitterFalloff; // emitter distance-falloff coefficient
    coneParams2Arr[3] = coneLightCount;
    device.queue.writeBuffer(coneShader.uniforms.params2.getGPUBuffer(device), 0, coneParams2Arr);

    coneAoArr[0] = probeParams.aoConeCount;
    coneAoArr[1] = probeParams.aoReach;
    coneAoArr[2] = probeParams.aoSteps;
    coneAoArr[3] = coneParams.emitterDirect; // emitter direct strength (vs the sun)
    device.queue.writeBuffer(coneShader.uniforms.aoParams.getGPUBuffer(device), 0, coneAoArr);

    mat4.invert(invViewProj, viewProjMatrix);
    coneInvArr.set(invViewProj as Float32Array);
    device.queue.writeBuffer(coneShader.uniforms.invViewProj.getGPUBuffer(device), 0, coneInvArr);

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: coneView,
          clearValue: [0, 0, 0, 1],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(conePipeline);
    pass.setBindGroup(0, coneGroup0);
    pass.draw(6, 1, 0, 0);
    pass.end();
  }

  // Upload the emitter data the cone pass importance-samples (aimed cones). `flat` = n*4 floats
  // (x,y,z,radius per light); `colorsFlat` = parallel n*4 (r,g,b,intensity per light) for the
  // analytic-direct shadow term. `count` is clamped to [0,8]; unused entries zeroed. The caller
  // discovers these from the LightEmitter component each frame (no manual light list). count=0 →
  // pure Fibonacci fill cones.
  function setLights(flat: number[], count: number, colorsFlat: number[]) {
    const n = Math.max(0, Math.min(8, count));
    coneLightsArr.fill(0);
    coneLightsArr.set(flat.slice(0, n * 4));
    device.queue.writeBuffer(coneShader.uniforms.lights.getGPUBuffer(device), 0, coneLightsArr);
    coneLightColorsArr.fill(0);
    coneLightColorsArr.set(colorsFlat.slice(0, n * 4));
    device.queue.writeBuffer(
      coneShader.uniforms.lightColor.getGPUBuffer(device),
      0,
      coneLightColorsArr,
    );
    coneLightCount = n;
  }

  // VCT composite (Layer 4): combine albedo + cone indirect/AO + direct sun + self-emission
  // into the final lit image → compositeOutput (full-res HDR). MUST run AFTER cone() (it reads
  // coneOutput). Reads the G-buffer (albedo/normal/depth/emission).
  function composite(encoder: GPUCommandEncoder) {
    compParamsArr[0] = compositeParams.ambient;
    compParamsArr[1] = compositeParams.exposure; // HDR exposure before ACES tonemap
    compParamsArr[2] = sunWorldTexel; // sun shadow-map world texel size (normal-offset bias)
    compParamsArr[3] = compositeParams.penumbra; // sun-dim → wider PCF (soft penumbra)
    device.queue.writeBuffer(
      compositeShader.uniforms.params.getGPUBuffer(device),
      0,
      compParamsArr,
    );

    // inverse(viewProj) for the sun-shadow world-position reconstruction.
    mat4.invert(invViewProj, viewProjMatrix);
    compInvArr.set(invViewProj as Float32Array);
    device.queue.writeBuffer(
      compositeShader.uniforms.invViewProj.getGPUBuffer(device),
      0,
      compInvArr,
    );

    compParams2Arr[0] = canvas.width;
    compParams2Arr[1] = canvas.height;
    compParams2Arr[2] = 0;
    compParams2Arr[3] = 0;
    device.queue.writeBuffer(
      compositeShader.uniforms.params2.getGPUBuffer(device),
      0,
      compParams2Arr,
    );

    // Directional sun: .xyz = world dir TOWARD the sun (azimuth + elevation), .w = effective
    // intensity (0 = disabled). Same packing as voxelize's uSun.
    const a = SunLight.angle;
    const e = SunLight.elevation;
    const ce = Math.cos(e);
    compSunArr[0] = Math.cos(a) * ce;
    compSunArr[1] = Math.sin(a) * ce;
    compSunArr[2] = Math.sin(e);
    compSunArr[3] = SunLight.enabled ? SunLight.intensity : 0;
    device.queue.writeBuffer(compositeShader.uniforms.sun.getGPUBuffer(device), 0, compSunArr);
    compSunColorArr[0] = SunLight.color[0];
    compSunColorArr[1] = SunLight.color[1];
    compSunColorArr[2] = SunLight.color[2];
    device.queue.writeBuffer(
      compositeShader.uniforms.sunColor.getGPUBuffer(device),
      0,
      compSunColorArr,
    );

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: compositeView,
          clearValue: [0, 0, 0, 1],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(compositePipeline);
    pass.setBindGroup(0, compositeGroup0);
    pass.draw(6, 1, 0, 0);
    pass.end();
  }

  // Change the voxel size (graininess). Destroys the old textures, rebuilds the grid.
  function setCellSize(newCellSize: number) {
    textures.voxelRadiance.destroy();
    buildGrid(newCellSize);
  }

  // Canvas resized: rebind the (new) G-buffer textures, recreate the canvas-sized cone +
  // composite outputs, and rebuild the cone/composite bind groups.
  function recreate(
    newDepth: GPUTexture,
    newNormal: GPUTexture,
    newAlbedo: GPUTexture,
    newEmission: GPUTexture,
  ) {
    gDepth = newDepth;
    gNormal = newNormal;
    gAlbedo = newAlbedo;
    gEmission = newEmission;
    coneOutput.destroy();
    coneOutput = createConeOutput();
    coneView = coneOutput.createView();
    compositeOutput.destroy();
    compositeOutput = createCompositeOutput();
    compositeView = compositeOutput.createView();
    // Bind groups below reference coneView, so refresh the cached views first.
    buildConeGroup();
    // Rebuild the composite group: G-buffer + coneOutput changed.
    buildCompositeGroup();
  }

  return {
    coneParams,
    compositeParams,
    probeParams,
    voxelize,
    mips,
    probe,
    cone,
    setLights,
    sunDepth,
    composite,
    recreate,
    setCellSize,
    get mipCount() {
      return mipCount;
    },
    get cellSize() {
      return cellSize;
    },
    get dims() {
      return { x: dimX, y: dimY, z: dimZ };
    },
    get textures() {
      return textures;
    },
    get coneOutputTexture() {
      return coneOutput;
    },
    get compositeOutputTexture() {
      return compositeOutput;
    },
    // Held for a future layer (Layer 4 reads albedo); exposed so the closure ref is live.
    get albedoTexture() {
      return gAlbedo;
    },
  };
}
