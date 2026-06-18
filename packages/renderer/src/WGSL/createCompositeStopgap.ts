import { GPUShader } from "./GPUShader.ts";
import { ShaderMeta } from "./ShaderMeta.ts";
import { VariableKind, VariableMeta } from "../Struct/VariableMeta.ts";
import { wgsl } from "./wgsl.ts";

/**
 * Phase 1 stopgap lighting. Throwaway pass that reads the G-buffer (albedo +
 * world normal + emission) and writes a simple Lambert-shaded image:
 *   lit = albedo * (ambient + saturate(dot(N, -sunDir))) + emission
 * so 3D impostors are *visible* before Radiance Cascades (Phase 2) replaces it.
 */
const shaderMeta = new ShaderMeta(
  {
    samp: new VariableMeta("samp", VariableKind.Sampler, `sampler`),
    albedo: new VariableMeta("albedo", VariableKind.Texture, `texture_2d<f32>`),
    normalTex: new VariableMeta("normalTex", VariableKind.Texture, `texture_2d<f32>`),
    emissionTex: new VariableMeta("emissionTex", VariableKind.Texture, `texture_2d<f32>`),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const POSITION = array<vec2f, 6>(
  vec2f(-1., -1.), vec2f(1., -1.), vec2f(1., 1.),
  vec2f(-1., -1.), vec2f(1., 1.), vec2f(-1., 1.));
const TEX = array<vec2f, 6>(
  vec2f(0., 1.), vec2f(1., 1.), vec2f(1., 0.),
  vec2f(0., 1.), vec2f(1., 0.), vec2f(0., 0.));

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VOut {
  var o: VOut;
  o.pos = vec4f(POSITION[i], 0., 1.);
  o.uv = TEX[i];
  return o;
}

const SUN_DIR: vec3f = normalize(vec3f(-0.4, -0.4, -0.8));
const AMBIENT: f32 = 0.35;

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let a = textureSample(albedo, samp, in.uv);
  if (a.a == 0.0) { discard; } // background: leave the swapchain clear value
  let n = normalize(textureSample(normalTex, samp, in.uv).xyz);
  let ndl = max(dot(n, -SUN_DIR), 0.0);
  let emission = textureSample(emissionTex, samp, in.uv).rgb;
  let lit = a.rgb * (AMBIENT + ndl) + emission;
  return vec4f(lit, 1.0);
}
  `,
);

export function createCompositeStopgap(device: GPUDevice) {
  const sampler = device.createSampler({
    magFilter: "nearest",
    minFilter: "nearest",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });

  let entry: { pipeline: GPURenderPipeline; bindGroup: GPUBindGroup } | undefined;

  function getEntry(albedo: GPUTexture, normal: GPUTexture, emission: GPUTexture) {
    if (entry === undefined) {
      shaderMeta.uniforms.samp.setSampler(sampler);
      shaderMeta.uniforms.albedo.setTexture(albedo);
      shaderMeta.uniforms.normalTex.setTexture(normal);
      shaderMeta.uniforms.emissionTex.setTexture(emission);
      const shader = new GPUShader(shaderMeta);
      entry = {
        pipeline: shader.getRenderPipeline(device, "vs_main", "fs_main", {
          targetFormat: "bgra8unorm",
        }),
        bindGroup: shader.getBindGroup(device, 0),
      };
    }
    return entry;
  }

  return function composite(
    encoder: GPUCommandEncoder,
    albedo: GPUTexture,
    normal: GPUTexture,
    emission: GPUTexture,
    out: GPUTexture,
  ) {
    const { pipeline, bindGroup } = getEntry(albedo, normal, emission);
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: out.createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6, 1, 0, 0);
    pass.end();
  };
}
