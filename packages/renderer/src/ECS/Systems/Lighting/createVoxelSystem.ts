import { getTypeTypedArray } from "../../../Shader/index.ts";
import { shaderMeta as voxelizeMeta } from "./stages/2_voxelize/voxelize.shader.ts";
import { DEFAULT_VOXEL_BAKED_CONFIG, type VoxelBakedConfig } from "./core/voxelConfig.ts";
import {
  createVoxelTextures,
  DEFAULT_VOXEL_GRID,
  voxelMipLevelCount,
  type VoxelGridConfig,
  type VoxelTextures,
} from "./core/voxelResources.ts";
import type { SceneInstances } from "../SDFSystem/createDrawShapeSystem.ts";
import { createVoxelizeSystem } from "./stages/2_voxelize/voxelizeSystem.ts";
import { createEmitterLightsSystem } from "./lights/emitterLightsSystem.ts";
import { createSunShadowSystem } from "./stages/1_sunShadow/sunShadowSystem.ts";
import { createMipPyramidSystem } from "./stages/3_mipPyramid/mipPyramidSystem.ts";
import { createAnisoVolumeSystem } from "./stages/4_anisoVolume/anisoVolumeSystem.ts";
import { createCompositeSystem } from "./stages/7_composite/compositeSystem.ts";
import { createScreenProbeSystem } from "./stages/5_screenProbe/screenProbeSystem.ts";
import { createConeSystem } from "./stages/6_cone/coneSystem.ts";
import { SunLight } from "../SunLight.ts";
import { cameraElevation, cameraPosition, cameraZoom } from "../ResizeSystem.ts";

// Voxel scene system: voxelize() fills the 3D albedo/emission/radiance textures from the SDF
// scene each frame; mips() builds the radiance pyramid; cone() gathers indirect light (N-cone
// VCT); sunDepth() renders the sun-POV shadow map; composite() produces the final lit image
// (see ./README.md for the full system scheme).
//
// GRANULARITY: the world box EXTENT is fixed, but its XY origin FOLLOWS THE CAMERA
// (updateGridOrigin — snapped so panning doesn't re-sample the field); cellSize controls voxel
// size (and thus per-axis dims = round(extent/cellSize)) — the "graininess" knob. Smaller
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

  // World box: the XY origin FOLLOWS THE CAMERA (updateGridOrigin below) so GI coverage is always
  // centered on what the player sees, and with autoCell the CELL SIZE follows the ZOOM (discrete
  // ×2 ladder) so the box always covers the screen with a constant voxel budget: texture dims are
  // the invariant, extent = dims × cellSize breathes with the ladder. Z origin stays fixed — the
  // camera orbits in XY. The BASE extents (from the initial config) are the reference the manual
  // buildGrid/setCellSize path derives dims from; the live extents track the current cell.
  let originX = grid.originX;
  let originY = grid.originY;
  const originZ = grid.originZ;
  const baseExtentX = grid.dimX * grid.cellSize;
  const baseExtentY = grid.dimY * grid.cellSize;
  const baseExtentZ = grid.dimZ * grid.cellSize;
  let extentX = baseExtentX;
  let extentY = baseExtentY;
  let extentZ = baseExtentZ;

  // LEVEL 1 — the coarse clipmap cascade: cell 2×, XY extent 2× (same XY dims), Z extent UNCHANGED
  // (the scene is flat — doubling Z would voxelize empty sky), so dimZ1 = dimZ/2. Its own camera-
  // snapped XY origin; Z shares the fine level's floor. Iso-only (no aniso set) with a full mip
  // chain of which the cone reads lods ≤ CLIP_L1_MAX_LOD.
  let origin1X = grid.originX - extentX * 0.5;
  let origin1Y = grid.originY - extentY * 0.5;

  // ===== Sub-systems (created once; only textures/bind groups rebuild). =====
  // Voxel-radiance mip pyramids — owned by their own sub-systems (downsample shader/pipeline +
  // per-level bind groups/buffers), one instance per clipmap level. buildGrid calls their
  // rebindGrid() after the radiance volumes are recreated; mips() runs both.
  const mipPyramid = createMipPyramidSystem({ device });
  const mipPyramid1 = createMipPyramidSystem({ device });

  // Anisotropic directional pyramid (the far-field anti-leak) — owned by its own sub-system (BASE +
  // VOLUME shaders/pipelines + the 6 directional volumes + per-level bind groups/buffers). buildGrid
  // calls its rebindGrid() after voxelRadiance is recreated; anisoBase()/anisoMips() delegate to its
  // base()/mips(); the screen-probe gather reads its 6 textures via getTextures().
  const anisoVolume = createAnisoVolumeSystem({ device });

  // Filtering sampler for textureSampleLevel over the rgba16float voxelRadiance pyramid + the aniso
  // directional volumes (the screen-probe SH textures are point-loaded, not sampled through this).
  const voxelSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    mipmapFilter: "linear",
  });

  // ===== Sun shadow map (depth-only pass from the sun's POV; grid/camera-independent). =====
  // Owned by its own sub-system: it renders the sun-POV depth map and computes the sun view-proj
  // matrix + world-texel size. The grid box is passed as an accessor so cellSize stays current
  // across setCellSize(). voxelize + composite read the matrix/texel/depth-view back through the
  // returned getters after sun.render() runs (see sunDepth() below).
  const sun = createSunShadowSystem({
    device,
    sceneInstances,
    getGridBox: () => ({ originX, originY, originZ, extentX, extentY, extentZ, cellSize }),
  });
  // VOXELIZE cluster — owns the voxelize shader/pipelines, the uPass buffers, the group-0/1 bind
  // groups (grid uniforms + scene buffers + the sun shadow map), the CPU AABB / dispatch scratch,
  // and issues the clear + two scatter passes. buildGrid() calls its rebindGrid() after the grid is
  // (re)built (rebinds the mip-0 storage target + refreshes the clear dispatch dims + grid uniforms);
  // voxelize() delegates to its run(). It reads the sun matrix/depth-view back through the sun getters.
  const voxelizeSys = createVoxelizeSystem({
    device,
    sceneInstances,
    getGridBox: () => ({ originX, originY, originZ, cellSize, dimX, dimY, dimZ }),
    getGridBox1: () => ({
      originX: origin1X,
      originY: origin1Y,
      originZ,
      cellSize: cellSize * 2,
      dimX: dim1X,
      dimY: dim1Y,
      dimZ: dim1Z,
    }),
    sun,
  });

  // --- Scratch typed arrays for uniform uploads. ---
  // Grid uniforms shared across the cone + gather shaders (the voxelize cluster owns its own copy).
  const originArr = getTypeTypedArray(voxelizeMeta.uniforms.gridOrigin.type); // Float32Array(4)
  const dimsArr = getTypeTypedArray(voxelizeMeta.uniforms.gridDims.type); // Int32Array(4)
  // Level-1 counterparts (gather-only readers; the cone's AO cones stay on level 0).
  const originArr1 = getTypeTypedArray(voxelizeMeta.uniforms.gridOrigin.type); // Float32Array(4)
  const dimsArr1 = getTypeTypedArray(voxelizeMeta.uniforms.gridDims.type); // Int32Array(4)
  // EMITTER LIGHTS (CPU clustered cull) sub-system. Owns the aimed-emitter storage buffer (uLights)
  // + the clustered-cull table (uLightClusters) and their CPU scratch. setLights() uploads the
  // emitters + refills/uploads the cluster table each frame; recreateLightClusters() resizes the
  // table to the current grid + baked clusterDiv/clusterCap. On a lightsBuf grow it rebuilds the
  // probe bind groups (they bind lightsBuf/clusterBuf via getters). buildGrid()/rebuild() delegate
  // to recreateLightClusters(); the gather's group 1 + uploadProbeUniforms read its getters.
  const emitterLights = createEmitterLightsSystem({
    device,
    config,
    getGridDims: () => ({ dimX, dimY, dimZ, originX, originY, originZ, cellSize }),
    onBuffersRecreated: () => screenProbe.rebuildGroups(),
  });
  // Runtime iso/aniso toggle for the cone pass (uParams2.z). Default on — the anti-leak is the point;
  // flip via setAnisoMode() (GUI) to A/B against the plain isotropic pyramid without a rebuild.
  let anisoMode = true;
  // Runtime clipmap toggle (gather uLightParams.z). Off = the exact old single-box path, for A/B.
  let clipmapMode = true;

  // --- Grid state (rebuilt by buildGrid). ---
  let cellSize = grid.cellSize;
  let dimX = grid.dimX;
  let dimY = grid.dimY;
  let dimZ = grid.dimZ;
  let dim1X = grid.dimX;
  let dim1Y = grid.dimY;
  let dim1Z = Math.max(1, grid.dimZ >> 1);
  let textures: VoxelTextures;
  // Level-1 volume pair (radiance pyramid + emission scatter target).
  let textures1: VoxelTextures;

  // (Re)build the voxel textures + the two texture-referencing bind groups for the
  // current cellSize, and upload the grid uniforms to all shaders.
  function buildGrid(newCellSize: number) {
    cellSize = newCellSize;
    // Dims derive from the BASE extents (the initial-config world box), not the live ones — the
    // zoom ladder scales the live extents without touching dims, and a manual rebuild must not
    // compound that scaling into the texture size.
    dimX = Math.max(1, Math.round(baseExtentX / cellSize));
    dimY = Math.max(1, Math.round(baseExtentY / cellSize));
    dimZ = Math.max(1, Math.round(baseExtentZ / cellSize));
    extentX = dimX * cellSize;
    extentY = dimY * cellSize;
    extentZ = dimZ * cellSize;
    // Level 1: cell 2×, same XY dims (→ XY extent 2×), Z extent unchanged (→ dimZ halved).
    const cell1 = cellSize * 2;
    dim1X = dimX;
    dim1Y = dimY;
    dim1Z = Math.max(1, dimZ >> 1);

    textures = createVoxelTextures(device, {
      originX,
      originY,
      originZ,
      dimX,
      dimY,
      dimZ,
      cellSize,
    });
    textures1 = createVoxelTextures(device, {
      originX: origin1X,
      originY: origin1Y,
      originZ,
      dimX: dim1X,
      dimY: dim1Y,
      dimZ: dim1Z,
      cellSize: cell1,
    });

    // Voxelize cluster: rebind both levels' storage targets (radiance mip 0s + the emitter
    // volumes) + refresh its clear dispatch dims + grid uniforms for the recreated volumes.
    voxelizeSys.rebindGrid(
      textures.voxelRadiance,
      textures.voxelEmission,
      textures1.voxelRadiance,
      textures1.voxelEmission,
      { originX, originY, originZ, cellSize, dimX, dimY, dimZ },
      {
        originX: origin1X,
        originY: origin1Y,
        originZ,
        cellSize: cell1,
        dimX: dim1X,
        dimY: dim1Y,
        dimZ: dim1Z,
      },
    );

    // Mip-pyramid downsample groups (mip L → L+1), one instance per clipmap level: each rebuilds
    // its per-level views/buffers for its recreated radiance volume + new dims.
    mipPyramid.rebindGrid(
      textures.voxelRadiance,
      dimX,
      dimY,
      dimZ,
      voxelMipLevelCount(dimX, dimY, dimZ),
    );
    mipPyramid1.rebindGrid(
      textures1.voxelRadiance,
      dim1X,
      dim1Y,
      dim1Z,
      voxelMipLevelCount(dim1X, dim1Y, dim1Z),
    );

    // ===== Anisotropic directional pyramid (rebuilt alongside the iso pyramid). =====
    // Delegated to the aniso sub-system: it recreates the 6 directional volumes (destroying the old
    // set), builds the BASE/VOLUME bind groups from iso voxelRadiance mip 0 + the new dims, and
    // uploads the per-level dims. Runs BEFORE buildScreenProbeGroups (which binds its textures).
    anisoVolume.rebindGrid(textures.voxelRadiance, {
      originX,
      originY,
      originZ,
      dimX,
      dimY,
      dimZ,
      cellSize,
    });

    // Grid uniforms (shared by all shaders). Populated BEFORE coneSys.rebindGrid() / screenProbe
    // .rebindGrid() so both sub-systems re-upload the fresh values to their own shader buffers.
    originArr[0] = originX;
    originArr[1] = originY;
    originArr[2] = originZ;
    originArr[3] = cellSize;
    dimsArr[0] = dimX;
    dimsArr[1] = dimY;
    dimsArr[2] = dimZ;
    dimsArr[3] = 0;
    originArr1[0] = origin1X;
    originArr1[1] = origin1Y;
    originArr1[2] = originZ;
    originArr1[3] = cell1;
    dimsArr1[0] = dim1X;
    dimsArr1[1] = dim1Y;
    dimsArr1[2] = dim1Z;
    dimsArr1[3] = 0;

    // Cone bind group references the rebuilt voxelRadiance view + the (stable) G-buffer; rebindGrid
    // also re-uploads the grid uniforms populated just above to the cone shader.
    coneSys.rebindGrid();
    // Composite bind group references the G-buffer (albedo/normal/emission/depth) + coneOutput.
    compositeSys.rebindGroup();
    // (composite no longer has grid uniforms — its sun shadow uses the shadow map, not voxels.)

    // Light-cluster buffer dims derive from the fresh grid dims — recreate BEFORE the probe
    // groups (their group 1 binds it).
    emitterLights.recreateLightClusters();
    // Screen-probe group0 references the rebuilt voxelRadiance view; it also uploads the screen
    // pass's gridOrigin/gridDims from the arrays populated just above. group2 references the
    // persistent screenProbeTex.
    screenProbe.rebindGrid();
  }

  // ===== Camera-following grid origin. =====
  // The box's XY origin tracks the camera look-at target (cameraPosition), SNAPPED to a multiple
  // of snapCells voxels. The snap is what keeps the move invisible: a snapped shift lands every
  // voxel center in the overlap on the SAME world points as last frame → bit-identical mip 0 →
  // no re-sampling shimmer while panning. But each mip L aggregates 2^L-cell blocks whose
  // PARTITION is anchored to the origin, so mips with blocks LARGER than the snap re-partition
  // on every snap step — the far-field cone contribution re-forms slightly (a "different
  // picture" pop; the cones reach LOD ~6 at maxDist 24 / aperture 0.577). snapCells trades the
  // two: bigger = more levels world-locked (16 locks iso mip ≤ 4 + aniso ≤ 3, leaving only the
  // blurriest tail to re-partition) but a coarser box step (16 cells = 8 wu at 0.5) — invisible
  // mid-scene, only the far coverage edge lurches. Live GUI knob; the REAL fix is clipmap
  // cascades replacing the deep mips (each level world-locked at its own granularity).
  //
  // Call sites: the head of setLights (so the CPU emitter clustering bins with the SAME origin
  // the gather shaders read this frame) and the head of renderFrame (for callers that drive
  // passes without setLights). Both run before any GPU pass reads uGridOrigin; the second call
  // in a frame is a no-op (cameraPosition is stable within a frame).
  let snapCells = 16;
  let followCamera = true;
  // Zoom ladder (autoCell): pick cellSize from the CURRENT zoom so the L0 box always covers the
  // visible ground footprint + an off-screen light margin, at constant texture dims. Discrete ×2
  // steps only — continuous scaling would re-sample the whole field every wheel tick (the same
  // no-snap-for-scaling argument as rotation); a step is a rare one-frame re-form the probe
  // hysteresis absorbs. Growing is immediate (coverage must not lag the screen); shrinking takes
  // a 15% hysteresis margin so the ladder can't oscillate at a threshold.
  let autoCell = true;
  // Floor = one step below the initial cell: extentZ = dimZ·cell must still cover the scene's
  // height (at the 0.5 default this floor is 0.25 → extentZ 16 wu, exactly the z ∈ [-2, 14] box).
  const CELL_LADDER_MIN = grid.cellSize / 2;
  const CELL_LADDER_MAX = grid.cellSize * 4;
  // Fraction of the visible ground DIAGONAL the L0 box must cover; the remaining corners + the
  // off-screen light margin ride level 1 (2× the extent). Calibrated so the DEFAULT demo view
  // (zoom 14, 256-dim grid) lands exactly on today's 0.5 cell — the ladder changes nothing until
  // the zoom actually leaves the band the current tuning was judged "fine" at.
  const L0_COVERAGE = 0.8;

  function zoomLadderCell(): number {
    // Visible half-extents in world units (mirrors updateProjectionMatrix): halfH = px/2/zoom;
    // the ground footprint stretches by 1/sin(elevation) along the view axis. The AABB of that
    // footprint rotates with the azimuth, so take the azimuth-INVARIANT bounding diagonal —
    // conservative (~√2), but orbiting can never force a ladder step.
    const halfH = canvas.offsetHeight / 2 / cameraZoom.value;
    const halfW = halfH * (canvas.offsetWidth / Math.max(1, canvas.offsetHeight));
    const elev = (cameraElevation.value * Math.PI) / 180;
    const groundHalfH = halfH / Math.max(0.2, Math.sin(elev));
    const halfDiag = Math.hypot(halfW, groundHalfH);
    const required = (2 * halfDiag * L0_COVERAGE) / Math.min(dimX, dimY);
    let c = cellSize;
    while (c < required && c < CELL_LADDER_MAX) c *= 2;
    while (c > CELL_LADDER_MIN && required <= c * 0.5 * 0.85) c *= 0.5;
    return c;
  }

  function updateGridOrigin() {
    if (!followCamera) return;
    // The ladder first: the snap quantum + extents below depend on the chosen cell. A cell step
    // needs NO texture rebuild — dims are unchanged, only uniforms/extents move (the volume is
    // re-voxelized every frame anyway).
    let cellChanged = false;
    if (autoCell) {
      const c = zoomLadderCell();
      if (c !== cellSize) {
        cellSize = c;
        extentX = dimX * c;
        extentY = dimY * c;
        extentZ = dimZ * c;
        originArr[3] = c;
        originArr1[3] = c * 2;
        cellChanged = true;
      }
    }
    const q = snapCells * cellSize;
    const ox = Math.round((cameraPosition.x - extentX * 0.5) / q) * q;
    const oy = Math.round((cameraPosition.y - extentY * 0.5) / q) * q;
    // Level 1 snaps to ITS OWN cell quantum (2× the fine one) around its own (2×) extent.
    const q1 = snapCells * cellSize * 2;
    const o1x = Math.round((cameraPosition.x - extentX) / q1) * q1;
    const o1y = Math.round((cameraPosition.y - extentY) / q1) * q1;
    if (!cellChanged && ox === originX && oy === originY && o1x === origin1X && o1y === origin1Y)
      return;
    originX = ox;
    originY = oy;
    origin1X = o1x;
    origin1Y = o1y;
    // Propagate to every shader holding a uGridOrigin/uGridOrigin1 copy. Uniform-buffer writes
    // only — dims and textures are unchanged, so NO bind groups rebuild. The voxelize cluster
    // re-uploads its own copies from getGridBox()/getGridBox1() at the head of every voxelize();
    // sunShadow + the emitter clustering read the live accessors each frame and need no push.
    originArr[0] = originX;
    originArr[1] = originY;
    originArr1[0] = origin1X;
    originArr1[1] = origin1Y;
    coneSys.uploadGridOrigin();
    screenProbe.uploadGridOrigin();
  }

  function setFollowCamera(on: boolean) {
    followCamera = on;
  }

  // Snap quantum in voxels (see the snapCells trade-off note above). Power of two keeps the
  // locked-mip reasoning exact.
  function setGridSnapCells(cells: number) {
    snapCells = Math.max(1, Math.round(cells));
  }

  // Zoom-ladder toggle. Off = the cell freezes at its current value (manual setCellSize regains
  // control); on = the ladder re-evaluates next updateGridOrigin.
  function setAutoCell(on: boolean) {
    autoCell = on;
  }

  // VCT composite (Layer 4 — the final lit image) sub-system. Owns the composite shader/pipeline +
  // bind group, the consolidated frame UBO, and the full-res HDR output texture; issues the composite
  // pass. It reads the G-buffer + coneOutput + the shared invViewProj + the sun matrix/texel through
  // the accessors below (cone() computes invViewProj before composite() runs). buildGrid()/recreate()
  // rebuild its group via rebindGroup()/resize(); rebuild() recompiles its shader.
  const compositeSys = createCompositeSystem({
    device,
    canvas,
    config,
    sun,
    voxelSampler,
    getGBuffer: () => ({ depth: gDepth, normal: gNormal, albedo: gAlbedo, emission: gEmission }),
    getConeView: () => coneSys.getOutputView(),
    getConeScale: () => coneSys.getConeScale(),
    getInvViewProj: () => coneSys.getInvViewProj(),
  });

  // SCREEN-PROBE cluster (the diffuse fill/bounce source: gather + adaptive placement + temporal +
  // budget + debug) sub-system. Owns the probe atlas ping-pong sets, the adaptive-atlas indirection
  // buffers, the two gather pipelines + the classify/refine/args placement pipelines + the debug
  // view, and all their bind groups + CPU scratch. Built AFTER anisoVolume + emitterLights +
  // compositeSys (its groups bind their textures/buffers/output view). buildGrid()/recreate()/rebuild()
  // delegate to its rebindGrid()/resize()/rebuild(); the cone pass reads its atlas + params via getters.
  const screenProbe = createScreenProbeSystem({
    device,
    canvas,
    config,
    originArr,
    dimsArr,
    originArr1,
    dimsArr1,
    getCellSize: () => cellSize,
    getVoxelRadiance: () => textures.voxelRadiance,
    getVoxelRadiance1: () => textures1.voxelRadiance,
    aniso: anisoVolume,
    getGBuffer: () => ({ depth: gDepth, normal: gNormal }),
    emitterLights,
    getAnisoMode: () => anisoMode,
    getClipmapMode: () => clipmapMode,
    getDebugTargetView: () => compositeSys.getOutputView(),
    onResourcesRecreated: () => coneSys.rebindGroups(),
    voxelSampler,
  });

  // CONE-RESOLVE cluster (Layer 2 — the screen-probe RESOLVE + short AO cones → half-res HDR)
  // sub-system. Owns the cone shader/pipeline, its two texture-referencing bind groups, the per-frame
  // uniform scratch, the shared reverse-Z inverse-VP mat4 (computed here once per frame — composite
  // reuses it), and the half-res HDR output. Built AFTER screenProbe (its group0 binds the atlas +
  // group1 binds the indirection buffers) and compositeSys (onOutputRecreated → its rebindGroup, since
  // composite samples coneView). buildGrid()/recreate()/rebuild() delegate to its rebindGrid()/resize()
  // /rebuild(); composite reads its output view / scale / invViewProj via getters.
  const coneSys = createConeSystem({
    device,
    canvas,
    config,
    screenProbe,
    getVoxelRadiance: () => textures.voxelRadiance,
    getGBuffer: () => ({ depth: gDepth, normal: gNormal }),
    voxelSampler,
    originArr,
    dimsArr,
    onOutputRecreated: () => compositeSys.rebindGroup(),
  });

  // Built last: buildGrid() (and recreate()) rebuild the composite group, which references
  // coneOutput, so that texture must exist first.
  buildGrid(cellSize);

  // Render the SDF scene from the sun's POV into the sun depth map (depth-only). MUST run after
  // prepare() (scene buffers current) and before composite() (which samples the map). The sun
  // sub-system refreshes the sun matrices each call, so it can run any time before composite.
  // The clusters that consume the sun view-proj it computed read it back through sun.getSunViewProj()
  // where they need it: voxelize() (samples the shadow map) uploads it at the head of its own pass;
  // composite() stages it into the frame UBO. Both run after this every frame, so reading it there
  // is byte-identical.
  function sunDepth(encoder: GPUCommandEncoder) {
    sun.render(encoder);
  }

  // Re-voxelize the scene into the 3D textures (run before debug()/the GI gather) — delegated to the
  // voxelize sub-system.
  function voxelize(encoder: GPUCommandEncoder) {
    voxelizeSys.voxelize(encoder);
  }

  // Build BOTH radiance mip pyramids — delegates to the per-level mip-pyramid sub-systems. Must
  // run AFTER voxelize() (each level 0 reads the mip 0 voxelize wrote) and in the SAME encoder.
  function mips(encoder: GPUCommandEncoder) {
    mipPyramid.run(encoder);
    mipPyramid1.run(encoder);
  }

  // Aniso BASE / VOLUME passes — delegate to the aniso sub-system.
  function anisoBase(encoder: GPUCommandEncoder) {
    anisoVolume.base(encoder);
  }

  function anisoMips(encoder: GPUCommandEncoder) {
    anisoVolume.mips(encoder);
  }

  // VCT cone GI: the screen-probe RESOLVE (fill/bounce + emitter light via the probe SH) + short
  // AO cones → coneOutput (HALF-res HDR; composite bilinear-upsamples). MUST run AFTER voxelize() +
  // mips() + screenProbe(). Reads the G-buffer (depth + normal). Delegated to the cone-resolve
  // sub-system (it also computes the shared invViewProj once per frame — composite reuses it).
  function cone(encoder: GPUCommandEncoder) {
    coneSys.cone(encoder);
  }

  // Explicit, infrequent action: recompile the three BAKED shaders (cone/composite/screen-probe)
  // with the CURRENT config, recreate their pipelines + bind groups, and re-upload the buildGrid-
  // time uniforms that the fresh GPU buffers lost (the per-frame ones refill next frame).
  function rebuild() {
    // Cone-resolve sub-system recompiles its shader + rebuilds its groups + re-uploads its grid
    // uniforms (originArr/dimsArr still hold the current grid values).
    coneSys.rebuild(config);
    // Composite sub-system recompiles its shader with the new baked config + rebuilds its group.
    compositeSys.rebuild(config);
    // The baked clusterDiv/clusterCap may have changed → the cluster buffer layout/size with them.
    emitterLights.recreateLightClusters();
    // Screen-probe sub-system recompiles its two gather shaders + rebuilds its groups (which also
    // re-uploads its gridOrigin/gridDims — originArr/dimsArr still hold the current grid values).
    screenProbe.rebuild(config);
  }

  // Change the voxel size (graininess). Destroys the old textures, rebuilds the grid.
  // (The aniso directional volumes are destroyed by anisoVolume.rebindGrid inside buildGrid.)
  function setCellSize(newCellSize: number) {
    // A manual cell choice is an override — stop the zoom ladder from re-deciding next frame.
    autoCell = false;
    textures.voxelRadiance.destroy();
    textures.voxelEmission.destroy();
    textures1.voxelRadiance.destroy();
    textures1.voxelEmission.destroy();
    buildGrid(newCellSize);
  }

  // Runtime toggle between the isotropic pyramid (false) and the anisotropic directional volumes
  // (true) — read next frame via the gather's uLightParams.y (the shader that owns the long cones).
  // No rebuild: A/B the anti-leak live.
  function setAnisoMode(on: boolean) {
    anisoMode = on;
  }

  // Runtime clipmap A/B: off = the gather ignores level 1 entirely (exact old single-box path).
  // The level-1 volumes keep being voxelized either way (their cost is the honest baseline).
  function setClipmapMode(on: boolean) {
    clipmapMode = on;
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
    // Screen-probe atlas + indirection buffers are canvas-derived → recreate them + rebuild the
    // screen-probe groups (group0 = G-buffer + voxelRadiance, group2 = the recreated screen textures).
    screenProbe.resize();
    // Cone-resolve sub-system: recreate its half-res output + rebuild its groups (which reference
    // coneView + the new G-buffer/screen textures — via screenProbe.getSpTex()/getProbeBufs()).
    coneSys.resize();
    // Composite sub-system: recreate its full-res output + rebuild its group (G-buffer + coneOutput
    // changed).
    compositeSys.resize();
  }

  // The full per-frame GI scenario, in the load-bearing order. The caller draws the SDF G-buffer
  // BEFORE this and calls present(compositeOutputTexture) AFTER; everything between is here so the
  // frame reads as one named call. sunDepth runs only when the directional sun is on; probeDebug
  // replaces composite when the debug view is toggled.
  function renderFrame(encoder: GPUCommandEncoder) {
    updateGridOrigin(); // camera-following box: must precede every pass that reads uGridOrigin
    if (SunLight.enabled) sunDepth(encoder); // sun-POV depth → voxelize injection + composite shadow
    voxelize(encoder); // scene → voxelRadiance mip 0
    mips(encoder); // isotropic radiance pyramid
    anisoBase(encoder); // 6 directional level-0 volumes
    anisoMips(encoder); // directional pyramids (far-field anti-leak)
    // Light-adaptive screen-probe atlas: clear → classify → gatherUniform → refine (16→8) →
    // build-args → gatherAdaptive. gatherUniform runs BEFORE refine so refine subdivides on the real
    // gathered-SH radiance spread across the cage.
    screenProbe.probeClear(encoder);
    screenProbe.probeClassify(encoder);
    screenProbe.gatherUniform(encoder);
    screenProbe.probeRefine(encoder);
    screenProbe.probeBuildArgs(encoder);
    screenProbe.gatherAdaptive(encoder);
    cone(encoder); // screen-probe resolve + AO → half-res HDR
    if (screenProbe.debugProbes)
      screenProbe.probeDebug(encoder); // debug view (same target as composite)
    else compositeSys.composite(encoder); // final lit image
  }

  return {
    config,
    rebuild,
    renderFrame,
    voxelize,
    mips,
    anisoBase,
    anisoMips,
    probeClear: screenProbe.probeClear,
    probeClassify: screenProbe.probeClassify,
    probeRefine: screenProbe.probeRefine,
    probeBuildArgs: screenProbe.probeBuildArgs,
    gatherUniform: screenProbe.gatherUniform,
    gatherAdaptive: screenProbe.gatherAdaptive,
    pollBudget: screenProbe.pollBudget,
    cone,
    // Origin update runs FIRST so the CPU emitter clustering bins with the same camera-following
    // origin the gather shaders will read this frame (no one-frame cluster/uniform mismatch).
    setLights: (data: Float32Array, count: number) => {
      updateGridOrigin();
      emitterLights.setLights(data, count);
    },
    sunDepth,
    composite: compositeSys.composite,
    probeDebug: screenProbe.probeDebug,
    recreate,
    setCellSize,
    setConeScale: coneSys.setConeScale,
    setAnisoMode,
    setClipmapMode,
    setFollowCamera,
    setGridSnapCells,
    setAutoCell,
    setScreenProbeTile: screenProbe.setScreenProbeTile,
    setScreenProbeParams: screenProbe.setScreenProbeParams,
    setAdaptiveFraction: screenProbe.setAdaptiveFraction,
    setRefineDiv: screenProbe.setRefineDiv,
    setLightThresh: screenProbe.setLightThresh,
    setScreenProbeResolveRadius: screenProbe.setScreenProbeResolveRadius,
    setTemporalHysteresis: screenProbe.setTemporalHysteresis,
    setDebugProbes: screenProbe.setDebugProbes,
    get anisoMode() {
      return anisoMode;
    },
    get followCamera() {
      return followCamera;
    },
    get gridSnapCells() {
      return snapCells;
    },
    get clipmapMode() {
      return clipmapMode;
    },
    get autoCell() {
      return autoCell;
    },
    get debugProbes() {
      return screenProbe.debugProbes;
    },
    get screenProbeTile() {
      return screenProbe.screenProbeTile;
    },
    get adaptiveFraction() {
      return screenProbe.adaptiveFraction;
    },
    get refineDiv1() {
      return screenProbe.refineDiv1;
    },
    get lightThresh() {
      return screenProbe.lightThresh;
    },
    get adaptiveProbeCount() {
      return screenProbe.adaptiveProbeCount;
    },
    get budgetExceeded() {
      return screenProbe.budgetExceeded;
    },
    get spNormalPow() {
      return screenProbe.spNormalPow;
    },
    get spPlaneK() {
      return screenProbe.spPlaneK;
    },
    get screenProbeResolveRadius() {
      return screenProbe.screenProbeResolveRadius;
    },
    get temporalHysteresis() {
      return screenProbe.temporalHysteresis;
    },
    get coneScale() {
      return coneSys.getConeScale();
    },
    get mipCount() {
      return mipPyramid.mipCount;
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
      return coneSys.getOutputTexture();
    },
    get compositeOutputTexture() {
      return compositeSys.getOutputTexture();
    },
    // Held for a future layer (Layer 4 reads albedo); exposed so the closure ref is live.
    get albedoTexture() {
      return gAlbedo;
    },
  };
}
