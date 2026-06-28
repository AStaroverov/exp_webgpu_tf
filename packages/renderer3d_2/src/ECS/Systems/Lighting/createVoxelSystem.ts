import { mat4, vec3 } from "gl-matrix";
import { GPUShader } from "../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../Shader/index.ts";
import { viewProjMatrix } from "../ResizeSystem.ts";
import { shaderMeta as voxelizeMeta, WORKGROUP, WORKGROUP_1D } from "./voxelize.shader.ts";
import { createConeShaderMeta } from "./voxelCone.shader.ts";
import { createCompositeShaderMeta } from "./voxelComposite.shader.ts";
import { shaderMeta as mipMeta, WORKGROUP as MIP_WG } from "./voxelMip.shader.ts";
import { createProbeShaderMeta, WORKGROUP as PROBE_WG } from "./voxelProbe.shader.ts";
import { createProbeBlurShaderMeta } from "./voxelProbeBlur.shader.ts";
import { shaderMeta as sunShadowMeta } from "./sunShadow.shader.ts";
import { DEFAULT_VOXEL_BAKED_CONFIG, type VoxelBakedConfig } from "./voxelConfig.ts";
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

// Half the shape's Z extent from its per-kind depth slot in Shape.values (stride 8).
// Mirrors footprint_half_z in sceneSDF.wgsl; sphere (6) is unhalved (radius == half-extent).
function footprintHalfZ(kind: number, values: Float32Array, k: number): number {
  switch (kind) {
    case 0:
      return values[k * 8 + 1] * 0.5;
    case 1:
      return values[k * 8 + 2] * 0.5;
    case 3:
    case 4:
      return values[k * 8 + 3] * 0.5;
    case 5:
      return values[k * 8 + 6] * 0.5;
    default:
      return values[k * 8 + 0];
  }
}

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
  // All quality/tuning knobs that are baked into the WGSL (cone/composite/probe shaders) live in
  // ONE config object. Mutate it and call rebuild() to recompile the affected shaders with the new
  // baked consts. Genuinely dynamic data (sun, camera, emitters, grid) stays in uniforms.
  const config: VoxelBakedConfig = { ...DEFAULT_VOXEL_BAKED_CONFIG };

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
  // Built from a factory that bakes the current config consts into the WGSL → reassignable on
  // rebuild().
  let coneShader = new GPUShader(createConeShaderMeta(config));
  let conePipeline = coneShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targetFormat: "rgba16float",
    withBlending: false,
  });
  // VCT composite (Layer 4): final = albedo·(ambient·AO + directSun + indirect) + emission.
  let compositeShader = new GPUShader(createCompositeShaderMeta(config));
  let compositePipeline = compositeShader.getRenderPipeline(device, "vs_main", "fs_main", {
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
  let probeShader = new GPUShader(createProbeShaderMeta(config));
  let probePipeline = probeShader.getComputePipeline(device, "main");
  let probeEmptyGroup1 = device.createBindGroup({
    layout: probeShader.createBindGroupLayout(device, 1),
    entries: [],
  });

  // Probe-volume blur: a SEPARABLE 3D Gaussian over the SH-L1 textures (config.probeBlurRadius),
  // run AFTER probe() so the cone fill samples a SPATIALLY-SMOOTHED bounce → a moving source's fill
  // stops stepping by probe cells without paying for more probe trace work. Three 1D passes (X/Y/Z,
  // O(R) taps each, NOT O(R³)) ping-pong A→B→A→B between the two SH sets. Baked R → rebuildable.
  let probeBlurShader = new GPUShader(createProbeBlurShaderMeta(config));
  let probeBlurPipelineX = probeBlurShader.getComputePipeline(device, "blur_x");
  let probeBlurPipelineY = probeBlurShader.getComputePipeline(device, "blur_y");
  let probeBlurPipelineZ = probeBlurShader.getComputePipeline(device, "blur_z");
  let probeBlurEmptyGroup1 = device.createBindGroup({
    layout: probeBlurShader.createBindGroupLayout(device, 1),
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
  // Second SH set: the probe-blur pass reads probeTextures (raw) and writes here; the cone pass
  // samples THIS blurred set. Same resolution as probeTextures (the blur is in-place in grid space).
  const probeTexturesBlur: ProbeTextures = createProbeTextures(device, DEFAULT_PROBE_DIMS);
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
  // Cone scratch. (params/aoParams/tune are now BAKED consts — no scratch arrays.)
  const coneParams2Arr = getTypeTypedArray(coneShader.shaderMeta.uniforms.params2.type); // Float32Array(4)
  const coneInvArr = getTypeTypedArray(coneShader.shaderMeta.uniforms.invViewProj.type); // Float32Array(16)
  // Probe scratch. (probeParams is now BAKED — no scratch array.)
  const probeOriginArr = getTypeTypedArray(probeShader.shaderMeta.uniforms.gridOrigin.type); // Float32Array(4)
  const probeGridDimsArr = getTypeTypedArray(probeShader.shaderMeta.uniforms.gridDims.type); // Int32Array(4)
  const probeDimsArr = getTypeTypedArray(probeShader.shaderMeta.uniforms.probeDims.type); // Int32Array(4)
  // Auto-discovered emitter centers the cone importance-samples (x,y,z,radius per light) +
  // parallel colors (r,g,b,intensity) for the analytic-direct shadow term.
  const coneLightsArr = getTypeTypedArray(coneShader.shaderMeta.uniforms.lights.type); // Float32Array(32)
  const coneLightColorsArr = getTypeTypedArray(coneShader.shaderMeta.uniforms.lightColor.type); // Float32Array(32)
  let coneLightCount = 0;
  // Composite scratch — ONE consolidated frame UBO (matches the WGSL `CompositeFrame` struct).
  // 48 f32 = 192 bytes. Field offsets (in f32 elements): params@0, params2@4, sun@8, sunColor@12,
  // invViewProj@16, sunViewProj@32. Filled across composite() + buildSunViewProj(), one writeBuffer.
  const compFrameArr = new Float32Array(48);
  const CF_PARAMS = 0;
  const CF_PARAMS2 = 4;
  const CF_SUN = 8;
  const CF_SUNCOLOR = 12;
  const CF_INVVP = 16;
  const CF_SUNVP = 32;
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
  // Camera-frustum fit scratch: inverse camera viewProj + a reused corner (unprojected NDC → world).
  const camInvViewProj = mat4.create();
  const camCorner = vec3.create();
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
  // Probe-blur bind groups for the separable ping-pong. Two src→dst directions, each a (group0 =
  // probeDims + src SH textures, group2 = dst SH textures) pair:
  //   AB: src = probeTextures (A)     → dst = probeTexturesBlur (B)   — used by the X and Z passes
  //   BA: src = probeTexturesBlur (B) → dst = probeTextures (A)       — used by the Y pass
  // They reference the persistent probe textures (resolution-fixed) → rebuilt only when the blur
  // shader recompiles (rebuild()), not on grid changes.
  let blurGroupAB0: GPUBindGroup;
  let blurGroupAB2: GPUBindGroup;
  let blurGroupBA0: GPUBindGroup;
  let blurGroupBA2: GPUBindGroup;
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
        coneShader.uniforms.params2.getBindGroupEntry(device),
        coneShader.uniforms.invViewProj.getBindGroupEntry(device),
        coneShader.uniforms.gridOrigin.getBindGroupEntry(device),
        coneShader.uniforms.gridDims.getBindGroupEntry(device),
        coneShader.uniforms.lights.getBindGroupEntry(device),
        coneShader.uniforms.lightColor.getBindGroupEntry(device),
        { binding: coneShader.shaderMeta.uniforms.depthTex.binding, resource: gDepth.createView() },
        { binding: coneShader.shaderMeta.uniforms.normalTex.binding, resource: gNormal.createView() },
        // ALL-mips sampled view so textureSampleLevel can pick any lod.
        {
          binding: coneShader.shaderMeta.uniforms.voxelRadiance.binding,
          resource: textures.voxelRadiance.createView({ dimension: "3d" }),
        },
        // Irradiance-probe SH-L1 volume (one sampled 3D view per channel) for the fill term.
        // Reads the BLURRED set (probeTexturesBlur), written by probeBlur() after probe().
        {
          binding: coneShader.shaderMeta.uniforms.shR.binding,
          resource: probeTexturesBlur.shR.createView({ dimension: "3d" }),
        },
        {
          binding: coneShader.shaderMeta.uniforms.shG.binding,
          resource: probeTexturesBlur.shG.createView({ dimension: "3d" }),
        },
        {
          binding: coneShader.shaderMeta.uniforms.shB.binding,
          resource: probeTexturesBlur.shB.createView({ dimension: "3d" }),
        },
        { binding: coneShader.shaderMeta.uniforms.voxelSampler.binding, resource: voxelSampler },
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
        {
          binding: probeShader.shaderMeta.uniforms.voxelRadiance.binding,
          resource: textures.voxelRadiance.createView({ dimension: "3d" }),
        },
        { binding: probeShader.shaderMeta.uniforms.voxelSampler.binding, resource: voxelSampler },
      ],
    });
    probeGroup2 = device.createBindGroup({
      layout: probePipeline.getBindGroupLayout(2),
      entries: [
        {
          binding: probeShader.shaderMeta.uniforms.shR.binding,
          resource: probeTextures.shR.createView({ dimension: "3d" }),
        },
        {
          binding: probeShader.shaderMeta.uniforms.shG.binding,
          resource: probeTextures.shG.createView({ dimension: "3d" }),
        },
        {
          binding: probeShader.shaderMeta.uniforms.shB.binding,
          resource: probeTextures.shB.createView({ dimension: "3d" }),
        },
      ],
    });
  }

  // (Re)build the probe-blur bind groups for both ping-pong directions. group0 = probeDims uniform
  // + the SOURCE SH textures; group2 = the DEST SH textures. All 3 entry points (blur_x/_y/_z)
  // share one pipeline layout, so any pipeline's layout works. Rebuilt when the blur shader is
  // recompiled (rebuild()).
  function buildBlurGroups() {
    // group0 = probeDims + the three source SH views.
    const group0 = (src: ProbeTextures) =>
      device.createBindGroup({
        layout: probeBlurPipelineX.getBindGroupLayout(0),
        entries: [
          probeBlurShader.uniforms.probeDims.getBindGroupEntry(device),
          {
            binding: probeBlurShader.shaderMeta.uniforms.srcR.binding,
            resource: src.shR.createView({ dimension: "3d" }),
          },
          {
            binding: probeBlurShader.shaderMeta.uniforms.srcG.binding,
            resource: src.shG.createView({ dimension: "3d" }),
          },
          {
            binding: probeBlurShader.shaderMeta.uniforms.srcB.binding,
            resource: src.shB.createView({ dimension: "3d" }),
          },
        ],
      });
    // group2 = the three destination SH storage views.
    const group2 = (dst: ProbeTextures) =>
      device.createBindGroup({
        layout: probeBlurPipelineX.getBindGroupLayout(2),
        entries: [
          {
            binding: probeBlurShader.shaderMeta.uniforms.dstR.binding,
            resource: dst.shR.createView({ dimension: "3d" }),
          },
          {
            binding: probeBlurShader.shaderMeta.uniforms.dstG.binding,
            resource: dst.shG.createView({ dimension: "3d" }),
          },
          {
            binding: probeBlurShader.shaderMeta.uniforms.dstB.binding,
            resource: dst.shB.createView({ dimension: "3d" }),
          },
        ],
      });
    blurGroupAB0 = group0(probeTextures); // src A
    blurGroupAB2 = group2(probeTexturesBlur); // dst B
    blurGroupBA0 = group0(probeTexturesBlur); // src B
    blurGroupBA2 = group2(probeTextures); // dst A
  }

  // Upload the blur pass's probeDims uniform (constant = the probe resolution). Called once after
  // the scratch arrays exist and again on rebuild() (the recompiled shader has a fresh buffer).
  function uploadBlurUniforms() {
    probeDimsArr[0] = probeDimsVal.x;
    probeDimsArr[1] = probeDimsVal.y;
    probeDimsArr[2] = probeDimsVal.z;
    probeDimsArr[3] = 0;
    device.queue.writeBuffer(probeBlurShader.uniforms.probeDims.getGPUBuffer(device), 0, probeDimsArr);
  }

  // (Re)build the Layer-4 composite bind group: uniforms + the G-buffer (albedo/normal/
  // emission) + the cone output (indirect+AO). The coneSampler bilinear-upsamples the half-res
  // cone output. Rebuilt when the G-buffer / coneOutput change (resize or grid rebuild).
  function buildCompositeGroup() {
    compositeGroup0 = device.createBindGroup({
      layout: compositePipeline.getBindGroupLayout(0),
      entries: [
        compositeShader.uniforms.frame.getBindGroupEntry(device),
        { binding: compositeShader.shaderMeta.uniforms.albedoTex.binding, resource: gAlbedo.createView() },
        { binding: compositeShader.shaderMeta.uniforms.normalTex.binding, resource: gNormal.createView() },
        { binding: compositeShader.shaderMeta.uniforms.coneTex.binding, resource: coneView },
        // Linear/clamp sampler (reuse voxelSampler) for the half-res cone upsample.
        { binding: compositeShader.shaderMeta.uniforms.coneSampler.binding, resource: voxelSampler },
        {
          binding: compositeShader.shaderMeta.uniforms.emissionTex.binding,
          resource: gEmission.createView(),
        },
        // Sun shadow: reverse-Z camera depth (reconstruct P) + the sun-POV depth map.
        { binding: compositeShader.shaderMeta.uniforms.depthTex.binding, resource: gDepth.createView() },
        { binding: compositeShader.shaderMeta.uniforms.shadowMap.binding, resource: sunDepthView },
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
  // DOWNSCALED HDR target for the Layer-2 cone gather. coneScale = 2 (half-res, ¼ the pixels →
  // ~4× less cone work) by default; 4 (quarter-res, 1/16 the pixels) for heavily-loaded scenes.
  // The composite normal-aware-upsamples it back to full res (indirect light is low-frequency, so
  // that is fine). The render pass viewport IS the texture size, so the cone pass renders at the
  // downscaled res automatically; the cone shader maps via texCoord, so it needs no scale uniform.
  let coneScale = 2;
  const createConeOutput = () =>
    device.createTexture({
      size: [
        Math.max(1, Math.ceil(canvas.width / coneScale)),
        Math.max(1, Math.ceil(canvas.height / coneScale)),
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

  // Probe-blur groups + uniform (independent of the grid → built once here, after the scratch
  // arrays exist; rebuilt only when the blur shader recompiles in rebuild()).
  buildBlurGroups();
  uploadBlurUniforms();

  // Compute the sun's orthographic view-projection (orthoZO, z in [0,1]). XY is fitted to the
  // CAMERA's visible region (clamped to the grid) so the 2048² shadow texels concentrate where the
  // camera looks → finer edges, and effective resolution scales with zoom (kills the texel
  // staircase). The DEPTH (Z) range spans the FULL grid so casters at any height are captured even
  // if they sit outside the view's XY (extending depth costs no XY resolution). Recomputed each
  // frame. Uploads sunViewProj to the sunShadow shader + composite, and rayDir to sunShadow.
  function buildSunViewProj() {
    // Grid AABB (world) — the clamp region + the depth range.
    const gMinX = originX;
    const gMinY = originY;
    const gMinZ = originZ;
    const gMaxX = originX + extentX;
    const gMaxY = originY + extentY;
    const gMaxZ = originZ + extentZ;

    // (A) Camera visible XY region: unproject the 8 reverse-Z NDC cube corners through the inverse
    // camera viewProj and clamp each into the grid box. The XY span of the clamped set is the
    // region the shadow map should cover at full resolution.
    mat4.invert(camInvViewProj, viewProjMatrix);
    let wMinX = Infinity;
    let wMinY = Infinity;
    let wMaxX = -Infinity;
    let wMaxY = -Infinity;
    for (let i = 0; i < 8; i++) {
      camCorner[0] = i & 1 ? 1 : -1;
      camCorner[1] = i & 2 ? 1 : -1;
      camCorner[2] = i & 4 ? 1 : 0; // reverse-Z: near=1, far=0 → both clip planes
      vec3.transformMat4(camCorner, camCorner, camInvViewProj);
      const px = Math.min(Math.max(camCorner[0], gMinX), gMaxX);
      const py = Math.min(Math.max(camCorner[1], gMinY), gMaxY);
      if (px < wMinX) wMinX = px;
      if (px > wMaxX) wMaxX = px;
      if (py < wMinY) wMinY = py;
      if (py > wMaxY) wMaxY = py;
    }
    // Degenerate (camera doesn't overlap the grid / inverted matrix) → fall back to the full grid.
    if (!(wMaxX > wMinX) || !(wMaxY > wMinY)) {
      wMinX = gMinX;
      wMaxX = gMaxX;
      wMinY = gMinY;
      wMaxY = gMaxY;
    }
    // Pad XY so a caster just outside the view still casts its shadow tip into it (the XY fit is
    // clipped; the margin trades a little resolution for fewer popping edges), re-clamped to grid.
    const padX = (wMaxX - wMinX) * 0.15 + cellSize;
    const padY = (wMaxY - wMinY) * 0.15 + cellSize;
    wMinX = Math.max(gMinX, wMinX - padX);
    wMaxX = Math.min(gMaxX, wMaxX + padX);
    wMinY = Math.max(gMinY, wMinY - padY);
    wMaxY = Math.min(gMaxY, wMaxY + padY);

    // Region center: XY from the fitted view region, Z from the full grid (eye centered in depth).
    const cx = (wMinX + wMaxX) * 0.5;
    const cy = (wMinY + wMaxY) * 0.5;
    const cz = (gMinZ + gMaxZ) * 0.5;

    // Direction TOWARD the sun (unit), same formula as voxelize()/composite().
    const a = SunLight.angle;
    const e = SunLight.elevation;
    const ce = Math.cos(e);
    const sdx = Math.cos(a) * ce;
    const sdy = Math.sin(a) * ce;
    const sdz = Math.sin(e);

    // Eye pulled back from the region center along +dirTowardSun; the ortho near/far fit below
    // folds the distance out, so any value past the bounding-sphere radius is fine.
    const radius = 0.5 * Math.hypot(wMaxX - wMinX, wMaxY - wMinY, extentZ);
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

    // Fit the ortho box IN LIGHT/VIEW space to the box [viewXY] × [full grid Z]: the full grid Z
    // is used for the corners' height so a tall caster's top (which under a tilted sun projects to
    // a different light-space XY than its base) is still inside the XY bounds. In view space the
    // camera looks down -Z, so visible z is negative.
    let lminX = Infinity;
    let lminY = Infinity;
    let lminZ = Infinity;
    let lmaxX = -Infinity;
    let lmaxY = -Infinity;
    let lmaxZ = -Infinity;
    for (let i = 0; i < 8; i++) {
      sunCorner[0] = i & 1 ? wMaxX : wMinX;
      sunCorner[1] = i & 2 ? wMaxY : wMinY;
      sunCorner[2] = i & 4 ? gMaxZ : gMinZ;
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
    // Composite's frame UBO is uploaded once in composite() (runs after sunDepth); here we just
    // stage the sun matrix + texel size into the shared scratch (params.z + the sunViewProj block).
    compFrameArr.set(sunViewProj as Float32Array, CF_SUNVP);
    compFrameArr[CF_PARAMS + 2] = sunWorldTexel;
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
    let prefix = 0;
    for (let k = 0; k < n; k++) {
      // Translation is column-major mat4 elements 12,13,14 (per-instance 16-float stride).
      const cx = tr[k * 16 + 12];
      const cy = tr[k * 16 + 13];
      const tz = tr[k * 16 + 14];
      const kind = kindArr[k];
      const round = roundArr[k];

      // Conservative bounding-circle radius in XY, computed from the SAME geometry the
      // footprint uses so the AABB never clips the shape. The depth slot must be EXCLUDED
      // from the XY bound, so each kind reads only its footprint (XY) slots — mirroring
      // footprint_half_xy in sceneSDF.wgsl.
      const v0 = valArr[k * 8 + 0];
      const v1 = valArr[k * 8 + 1];
      const v2 = valArr[k * 8 + 2];
      let rxyShape: number;
      if (kind === 3) {
        // Parallelogram: skew widens the X half-extent (halfX = width/2 + |skew|); the worst
        // corner is at hypot(halfX, height/2). values = (width, height, skew, depth).
        rxyShape = Math.hypot(v0 / 2 + Math.abs(v2), v1 / 2);
      } else if (kind === 5) {
        // Triangle: the first 6 slots are signed vertex coords (ax,ay,bx,by,cx,cy). The
        // conservative radius is the farthest vertex distance from the local origin.
        rxyShape = Math.max(
          Math.hypot(valArr[k * 8 + 0], valArr[k * 8 + 1]),
          Math.hypot(valArr[k * 8 + 2], valArr[k * 8 + 3]),
          Math.hypot(valArr[k * 8 + 4], valArr[k * 8 + 5]),
        );
      } else if (kind === 0 || kind === 6) {
        // Circle/cylinder + sphere: values[0] = radius (the full XY half-extent).
        rxyShape = v0;
      } else if (kind === 4) {
        // Trapezoid: values = [topWidth, bottomWidth, ySize, depth]. Bound = wider end / 2 in X,
        // ySize / 2 in Y → the worst corner is at hypot of those.
        rxyShape = Math.hypot(Math.max(v0, v1) / 2, v2 / 2);
      } else {
        // Rectangle/box: values = [width, height, depth]. Corner at hypot(width/2, height/2).
        rxyShape = Math.hypot(v0 / 2, v1 / 2);
      }
      const rxy = rxyShape + round + cellSize;
      // Z half-extent: sphere (kind 6) uses its radius (values[0]); everything else is half
      // the per-kind depth slot. Expanded by roundness + one cell.
      const zHalf = footprintHalfZ(kind, valArr, k) + round + cellSize;
      const centerZ = tz;

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

    // Upload the AABB lists. Only the live n entries matter (the binary search bound is
    // uInstanceCount = n), so upload exactly n*4 i32 elements instead of the whole MAX-sized buffer.
    device.queue.writeBuffer(voxShader.uniforms.aabbMin.getGPUBuffer(device), 0, aabbMinArr, 0, n * 4);
    device.queue.writeBuffer(voxShader.uniforms.aabbDim.getGPUBuffer(device), 0, aabbDimArr, 0, n * 4);

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
    // conesPerProbe / maxDist / aperture are now BAKED consts in the probe shader — nothing to
    // upload per frame; just dispatch.
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

  // Probe-volume blur: SEPARABLE 3D Gaussian over the SH-L1 set. MUST run AFTER probe() (reads its
  // output) and BEFORE cone() (which samples the blurred set). Three 1D compute passes ping-pong
  // A→B (X), B→A (Y), A→B (Z) so the FINAL blurred volume lands in probeTexturesBlur (B) — exactly
  // what buildConeGroup binds. Each pass is a separate compute pass so the encoder barriers between
  // them (a pass reads the previous pass's writes). With probeBlurRadius=0 each pass is a copy.
  const blurDX = Math.ceil(probeDimsVal.x / PROBE_WG);
  const blurDY = Math.ceil(probeDimsVal.y / PROBE_WG);
  const blurDZ = Math.ceil(probeDimsVal.z / PROBE_WG);
  function blurPass(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    group0: GPUBindGroup,
    group2: GPUBindGroup,
  ) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group0);
    pass.setBindGroup(1, probeBlurEmptyGroup1);
    pass.setBindGroup(2, group2);
    pass.dispatchWorkgroups(blurDX, blurDY, blurDZ);
    pass.end();
  }
  function probeBlur(encoder: GPUCommandEncoder) {
    blurPass(encoder, probeBlurPipelineX, blurGroupAB0, blurGroupAB2); // X: A→B
    blurPass(encoder, probeBlurPipelineY, blurGroupBA0, blurGroupBA2); // Y: B→A
    blurPass(encoder, probeBlurPipelineZ, blurGroupAB0, blurGroupAB2); // Z: A→B (final in B)
  }

  // VCT cone GI: per-pixel AIMED emitter cones (sharp direct + shadow) + a trilinear probe-SH
  // fetch for the fill/bounce + short AO cones → coneOutput (HALF-res HDR; composite bilinear-
  // upsamples). MUST run AFTER voxelize() + mips() + probe(). Reads the G-buffer (depth + normal).
  function cone(encoder: GPUCommandEncoder) {
    // params/aoParams/tune are BAKED consts now — only params2 carries dynamic data.
    // .x = canvas width, .y = canvas height, .z = SPARE (emitterFalloff is baked), .w = light count.
    coneParams2Arr[0] = canvas.width;
    coneParams2Arr[1] = canvas.height;
    coneParams2Arr[2] = 0;
    coneParams2Arr[3] = coneLightCount;
    device.queue.writeBuffer(coneShader.uniforms.params2.getGPUBuffer(device), 0, coneParams2Arr);

    // invViewProj computed ONCE per frame here (cone runs before composite, which reuses it).
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
    // Fill the consolidated frame UBO and upload it in ONE writeBuffer. params.z + the sunViewProj
    // block (CF_SUNVP) are already staged by buildSunViewProj() in sunDepth() (runs before this).
    // ambient/exposure/penumbra are BAKED consts; params.z = sun shadow-map world texel size.
    compFrameArr[CF_PARAMS + 2] = sunWorldTexel;
    // invViewProj already computed in cone() this frame (cone runs before composite); just reuse it.
    compFrameArr.set(invViewProj as Float32Array, CF_INVVP);
    compFrameArr[CF_PARAMS2 + 0] = canvas.width;
    compFrameArr[CF_PARAMS2 + 1] = canvas.height;
    compFrameArr[CF_PARAMS2 + 2] = coneScale; // cone downscale factor → upsample_cone maps cone↔full res
    compFrameArr[CF_PARAMS2 + 3] = 0;
    // Directional sun: .xyz = world dir TOWARD the sun (azimuth + elevation), .w = effective
    // intensity (0 = disabled). Same packing as voxelize's uSun.
    const a = SunLight.angle;
    const e = SunLight.elevation;
    const ce = Math.cos(e);
    compFrameArr[CF_SUN + 0] = Math.cos(a) * ce;
    compFrameArr[CF_SUN + 1] = Math.sin(a) * ce;
    compFrameArr[CF_SUN + 2] = Math.sin(e);
    compFrameArr[CF_SUN + 3] = SunLight.enabled ? SunLight.intensity : 0;
    compFrameArr[CF_SUNCOLOR + 0] = SunLight.color[0];
    compFrameArr[CF_SUNCOLOR + 1] = SunLight.color[1];
    compFrameArr[CF_SUNCOLOR + 2] = SunLight.color[2];
    device.queue.writeBuffer(compositeShader.uniforms.frame.getGPUBuffer(device), 0, compFrameArr);

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

  // Explicit, infrequent action: recompile the three BAKED shaders (cone/composite/probe) with the
  // CURRENT config, recreate their pipelines + bind groups, and re-upload the buildGrid-time
  // uniforms that the fresh GPU buffers lost (the per-frame ones refill next frame).
  function rebuild() {
    coneShader.destroy();
    compositeShader.destroy();
    probeShader.destroy();
    coneShader = new GPUShader(createConeShaderMeta(config));
    compositeShader = new GPUShader(createCompositeShaderMeta(config));
    probeShader = new GPUShader(createProbeShaderMeta(config));
    conePipeline = coneShader.getRenderPipeline(device, "vs_main", "fs_main", {
      targetFormat: "rgba16float",
      withBlending: false,
    });
    compositePipeline = compositeShader.getRenderPipeline(device, "vs_main", "fs_main", {
      targetFormat: "rgba16float",
      withBlending: false,
    });
    probePipeline = probeShader.getComputePipeline(device, "main");
    probeEmptyGroup1 = device.createBindGroup({
      layout: probeShader.createBindGroupLayout(device, 1),
      entries: [],
    });
    // Probe-blur shader is ALSO baked (probeBlurRadius) → recompile + rebuild its groups/uniform.
    probeBlurShader.destroy();
    probeBlurShader = new GPUShader(createProbeBlurShaderMeta(config));
    probeBlurPipelineX = probeBlurShader.getComputePipeline(device, "blur_x");
    probeBlurPipelineY = probeBlurShader.getComputePipeline(device, "blur_y");
    probeBlurPipelineZ = probeBlurShader.getComputePipeline(device, "blur_z");
    probeBlurEmptyGroup1 = device.createBindGroup({
      layout: probeBlurShader.createBindGroupLayout(device, 1),
      entries: [],
    });
    buildConeGroup();
    buildCompositeGroup();
    buildProbeGroups();
    buildBlurGroups();
    uploadBlurUniforms();
    // Re-upload buildGrid-time uniforms to the NEW shader buffers (per-frame ones refill next frame).
    device.queue.writeBuffer(coneShader.uniforms.gridOrigin.getGPUBuffer(device), 0, originArr);
    device.queue.writeBuffer(coneShader.uniforms.gridDims.getGPUBuffer(device), 0, dimsArr);
    device.queue.writeBuffer(probeShader.uniforms.gridOrigin.getGPUBuffer(device), 0, probeOriginArr);
    device.queue.writeBuffer(probeShader.uniforms.gridDims.getGPUBuffer(device), 0, probeGridDimsArr);
    device.queue.writeBuffer(probeShader.uniforms.probeDims.getGPUBuffer(device), 0, probeDimsArr);
  }

  // Change the voxel size (graininess). Destroys the old textures, rebuilds the grid.
  function setCellSize(newCellSize: number) {
    textures.voxelRadiance.destroy();
    buildGrid(newCellSize);
  }

  // Change the cone-pass downscale factor (2 = half-res, 4 = quarter-res). Recreates the cone
  // output at the new size and rebuilds the composite bind group (which samples it). The cone
  // shader is resolution-agnostic; the composite reads coneScale via its params2 each frame.
  function setConeScale(newScale: number) {
    coneScale = Math.max(1, Math.round(newScale));
    coneOutput.destroy();
    coneOutput = createConeOutput();
    coneView = coneOutput.createView();
    buildCompositeGroup(); // composite samples coneView → rebind the new texture
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
    config,
    rebuild,
    voxelize,
    mips,
    probe,
    probeBlur,
    cone,
    setLights,
    sunDepth,
    composite,
    recreate,
    setCellSize,
    setConeScale,
    get coneScale() {
      return coneScale;
    },
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
