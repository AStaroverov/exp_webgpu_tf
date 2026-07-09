import { mat4 } from "gl-matrix";
import { GPUShader } from "../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../Shader/index.ts";
import { viewProjMatrix } from "../ResizeSystem.ts";
import { shaderMeta as sunShadowMeta } from "./sunShadow.shader.ts";
import { SunLight } from "../SunLight.ts";
import { createSunViewProjComputer, type SunViewProjBox } from "./sunViewProj.ts";
import type { SceneInstances } from "../SDFSystem/createDrawShapeSystem.ts";

// Sun shadow map (depth-only pass from the sun's POV; grid/camera-independent).
// Standard depth (orthoZO [0,1]) so the composite's shadow test is the simple "fragment
// depth > stored ⇒ shadowed". Pipeline uses depthCompare "less-equal" + clear 1.0, fully
// decoupled from the main camera's reverse-Z. Built ONCE; only sunViewProj/rayDir refresh.
// 2048² over the 64-unit world box ≈ 0.03 world/texel — crisp enough for small objects, and
// 4× cheaper to render + store than 4096² (depth32float: 4096²=67 MB → 2048²=17 MB). The map
// re-renders every frame because the scene is dynamic (its content depends on object positions,
// not just the sun direction — so it cannot be cached across frames while objects move).
//
// This module OWNS the sunShadow shader's own uploads (uViewProj, uRayDir) and the depth render.
// The sun view-proj matrix + world-texel size it computes are ALSO consumed by other clusters
// (voxelize samples the shadow map, composite uses the matrix + texel bias); those clusters read
// them back through getSunViewProj()/getSunWorldTexel() after render() runs.
export function createSunShadowSystem({
  device,
  sceneInstances,
  getGridBox,
}: {
  device: GPUDevice;
  sceneInstances: SceneInstances;
  // The grid box (origin + extent + cellSize). Passed as an accessor so cellSize stays current
  // across setCellSize().
  getGridBox: () => SunViewProjBox;
}) {
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

  // Sun shadow scratch (allocate ONCE — never per frame). sunViewProj is computed each frame by
  // computeSun (which owns all the matrix/vector scratch) from SunLight + the grid AABB, then
  // uploaded to the sunShadow shader (vs uViewProj). rayDir = sun travel direction (= -dirTowardSun).
  const computeSun = createSunViewProjComputer();
  const sunViewProj = mat4.create();
  const sunViewProjArr = getTypeTypedArray(sunShadowMeta.uniforms.viewProj.type); // Float32Array(16)
  const sunRayDirArr = getTypeTypedArray(sunShadowMeta.uniforms.rayDir.type); // Float32Array(4)
  // World units per shadow texel (sun ortho width / SHADOW_SIZE) → composite normal-offset bias.
  let sunWorldTexel = 0;

  // Compute the sun's orthographic view-projection (orthoZO, z in [0,1]). XY is fitted to the
  // CAMERA's visible region (clamped to the grid) so the 2048² shadow texels concentrate where the
  // camera looks → finer edges, and effective resolution scales with zoom (kills the texel
  // staircase). The DEPTH (Z) range spans the FULL grid so casters at any height are captured even
  // if they sit outside the view's XY (extending depth costs no XY resolution). Recomputed each
  // frame. Uploads sunViewProj + rayDir to the sunShadow shader (its own uniforms).
  function buildSunViewProj() {
    // Pure matrix fit → sunViewProj (see sunViewProj.ts); the uploads stay here.
    const r = computeSun(getGridBox(), SHADOW_SIZE, viewProjMatrix, SunLight, sunViewProj);
    sunWorldTexel = r.sunWorldTexel;

    sunViewProjArr.set(sunViewProj as Float32Array);
    device.queue.writeBuffer(
      sunShadowShader.uniforms.viewProj.getGPUBuffer(device),
      0,
      sunViewProjArr,
    );

    // Sun travel direction = -dirTowardSun (already unit). xyz dir, w unused.
    sunRayDirArr[0] = -r.sdx;
    sunRayDirArr[1] = -r.sdy;
    sunRayDirArr[2] = -r.sdz;
    sunRayDirArr[3] = 0;
    device.queue.writeBuffer(sunShadowShader.uniforms.rayDir.getGPUBuffer(device), 0, sunRayDirArr);
  }

  // Render the SDF scene from the sun's POV into sunDepthTexture (depth-only). MUST run after
  // prepare() (scene buffers current) and before composite() (which samples the map). Refreshes
  // the sun matrices each call, so it can run any time before composite.
  function render(encoder: GPUCommandEncoder) {
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

  return {
    render,
    // The sun view-proj matrix that rendered THIS frame's map — consumed by voxelize (samples the
    // shadow map) and composite (shadow test). Valid after render() runs.
    getSunViewProj() {
      return sunViewProj;
    },
    // World units per shadow texel → composite's normal-offset shadow bias. Valid after render().
    getSunWorldTexel() {
      return sunWorldTexel;
    },
    // The sun-POV depth texture view — bound as the shadow-map sample source by voxelize + composite.
    getDepthView() {
      return sunDepthView;
    },
  };
}
