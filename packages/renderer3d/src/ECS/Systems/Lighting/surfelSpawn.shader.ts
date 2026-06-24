import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { SURFEL_CAP } from "./surfelResources.ts";

// Surfel Radiance Cascades — Stage A SPAWN compute pass.
//
// One thread per G-buffer pixel (workgroup 8x8, dispatch ceil(W/8) x ceil(H/8)).
// Each thread reconstructs the pixel's world position from the reverse-Z depth +
// inverse view-projection (copied verbatim from worldComposite.shader.ts), reads
// the G-buffer normal, and — gated by a low per-pixel random chance so surfels
// accumulate gradually — atomically allocates a surfel slot and writes its
// position(+radius²) and normal. No coverage test, no recycle, no grid yet
// (those are stages B/C/D): surfels just accumulate up to CAP and stay.
//
// COORDINATES: world is Z-up, ORTHOGRAPHIC tilted camera, reverse-Z (NEAR=1,
// FAR=0). depth <= 0 => far/background => no surface. Surfel radius is a FIXED
// world-unit value (no perspective fov term), squared into posr.w.
//
// The three surfel buffers (surfel_stack/posr/norw) are STANDALONE GPUBuffers
// (see surfelResources.ts) — declared here as StorageWrite VariableMetas ONLY for
// WGSL emission + the (kind-based, size-agnostic) bind-group layout entry. Their
// `atomic<u32>` / sized-array types are never parsed for size, so the JS type-size
// parser is never invoked for them. Bind them manually against
// pipeline.getBindGroupLayout(2).

// COMPUTE-visibility group-0 binding (uniforms + textures must advertise the
// compute stage, else the dispatch fails layout validation).
const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, {
    group: 0,
    visibility: GPUShaderStage.COMPUTE,
  });

export const shaderMeta = new ShaderMeta(
  {
    // ---- group 0 : per-frame uniforms (COMPUTE-only) ----
    // Inverse view-projection (reverse-Z) for world reconstruction; matches
    // ResizeSystem.viewProjMatrix inverse, exactly like worldComposite.
    invViewProj: uC("uInvViewProj", `mat4x4<f32>`),
    // Packed scalars A: .x = resW, .y = resH, .z = frameIndex (as f32), .w = spawnChance.
    packA: uC("uPackA", `vec4<f32>`),
    // Packed scalars B: .x = surfelRadius (world units), .y = surfelCap (as f32),
    // .z/.w unused.
    packB: uC("uPackB", `vec4<f32>`),

    // ---- group 0 : G-buffer textures (COMPUTE-only) ----
    // Reverse-Z depth (depth32float). sampleType MUST be "depth" — depth32float is
    // not filterable, so the layout entry has to advertise Depth.
    depthTexture: new VariableMeta("depthTexture", VariableKind.Texture, `texture_depth_2d`, {
      group: 0,
      visibility: GPUShaderStage.COMPUTE,
      textureSampleType: "depth",
    }),
    // G-buffer world normals (packed *0.5+0.5; a = surface mask).
    normalTexture: new VariableMeta("normalTexture", VariableKind.Texture, `texture_2d<f32>`, {
      group: 0,
      visibility: GPUShaderStage.COMPUTE,
    }),

    // ---- group 2 : surfel storage (StorageWrite => @group(2)) ----
    // STANDALONE buffers (surfelResources.ts) — these metas are layout/WGSL-emission
    // only. surfel_stack: [0] = atomic allocated count, [1+i] = free-id pool.
    surfelStack: new VariableMeta(
      "surfel_stack",
      VariableKind.StorageWrite,
      `array<atomic<u32>, ${SURFEL_CAP + 1}>`,
      { group: 2, binding: 0, visibility: GPUShaderStage.COMPUTE },
    ),
    // xyz = world position, w = radius² (w == 0 => DEAD slot).
    surfelPosr: new VariableMeta(
      "surfel_posr",
      VariableKind.StorageWrite,
      `array<vec4<f32>, ${SURFEL_CAP}>`,
      { group: 2, binding: 1, visibility: GPUShaderStage.COMPUTE },
    ),
    // xyz = world normal, w = recycle marker (set 2.0 on spawn).
    surfelNorw: new VariableMeta(
      "surfel_norw",
      VariableKind.StorageWrite,
      `array<vec4<f32>, ${SURFEL_CAP}>`,
      { group: 2, binding: 2, visibility: GPUShaderStage.COMPUTE },
    ),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
// ===== hash / random (self-contained wang/PCG; spawn gate) =====
fn wang_hash(seed_in: u32) -> u32 {
  var seed = seed_in;
  seed = (seed ^ 61u) ^ (seed >> 16u);
  seed = seed * 9u;
  seed = seed ^ (seed >> 4u);
  seed = seed * 0x27d4eb2du;
  seed = seed ^ (seed >> 15u);
  return seed;
}
fn random01(seed: u32) -> f32 {
  return f32(wang_hash(seed)) / 4294967295.0;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.xy;
  let resW = u32(uPackA.x);
  let resH = u32(uPackA.y);
  // 1. Bounds check (dispatch is ceil-rounded; drop the over-edge threads).
  if (px.x >= resW || px.y >= resH) {
    return;
  }

  let res = vec2<f32>(uPackA.x, uPackA.y);

  // 2. Reject background: reverse-Z => depth == 0 is far. (Copied from worldComposite.)
  let depth = textureLoad(depthTexture, vec2<i32>(px), 0);
  if (depth <= 0.0) {
    return;
  }

  // 3. Reject no-surface pixels (normal alpha is the G-buffer surface mask).
  let n = textureLoad(normalTexture, vec2<i32>(px), 0);
  if (n.a < 0.5) {
    return;
  }

  // 4. Reconstruct world position (reverse-Z + inverse VP; verbatim worldComposite).
  let uv = (vec2<f32>(px) + 0.5) / res;
  let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
  let wp4 = uInvViewProj * ndc;
  let world = wp4.xyz / wp4.w;

  // 5. Unpack the world-space normal (stored *0.5+0.5).
  let normal = normalize(n.rgb * 2.0 - 1.0);

  // 6. Random gate: low per-pixel chance so surfels fill in gradually, not CAP in
  //    one frame. seed mixes the pixel and the frame index.
  let frameIndex = u32(uPackA.z);
  let seed = px.x + px.y * resW + frameIndex * resW * resH;
  if (random01(seed) > uPackA.w) {
    return;
  }

  // 7. Allocate a surfel slot from the free-id stack.
  let cap = u32(uPackB.y);
  let ptr = atomicAdd(&surfel_stack[0], 1u);
  if (ptr >= cap) {
    // Pool exhausted — leave the count saturated (no recycle in Stage A) and bail.
    return;
  }
  let id = atomicLoad(&surfel_stack[1u + ptr]);

  // 8. Write the surfel: posr.w = radius² (>0 => live), norw.w = recycle marker.
  let radius = uPackB.x;
  surfel_posr[id] = vec4<f32>(world, radius * radius);
  surfel_norw[id] = vec4<f32>(normal, 2.0);
}
    `,
);
