import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { WORLD_DIR0_W, WORLD_GRID_DIM } from "./worldGather.shader.ts";

// Stage-1 world-space Radiance Cascades — DIRECT COMPOSITE pass (doc §7).
//
// One cascade (c0), no merge. Fullscreen draw over the canvas-sized worldLitTexture.
// For each screen pixel we:
//   1. Reconstruct world position from the reverse-Z depth buffer + the inverse
//      view-projection (matching ResizeSystem.viewProjMatrix exactly, the same way
//      overlay.shader.ts projects with the forward matrix — here we go the other way).
//   2. Read the G-buffer normal (packed *0.5+0.5, a = surface mask). a < 0.5 means
//      no surface (background) -> passthrough the scene * ambient (no probe light).
//   3. Bilinearly pick the 4 probes of cascade-0 around world.xy in the SINGLE probe
//      atlas (probeRad). Each probe owns a DIR0_W x DIR0_W octahedral tile.
//   4. Integrate the tile directions weighted by max(0, dot(N, dir)) (Lambert), with
//      solid-angle normalization 4*PI / (DIR0_W*DIR0_W), blend the 4 probes bilinearly.
//   5. lit = albedo * (ambient + radiance). The directional bonus stays consistent
//      with overlay.shader.ts (an additive, never-darkening dirGain term).
//
// COORDINATES: world is Z-up, footprints in XY. probeRad probe (i,j) lives at world
// XY probe_world_xy(ij) = uGridOrigin.xy + (ij + 0.5 - GRID_DIM/2) * uCell0 on plane
// uProbePlaneZ — IDENTICAL to the gather pass, so both agree on the same grid.

// Atlas/grid constants — MUST match worldGather.shader.ts and the probeRad texture.
const GRID_DIM = WORLD_GRID_DIM;
const DIR0_W = WORLD_DIR0_W;

export const shaderMeta = new ShaderMeta(
  {
    // ---- group 0 : textures + sampler ----
    inputSampler: new VariableMeta("textureSampler", VariableKind.Sampler, `sampler`),
    // Albedo / main pass color the light multiplies over.
    sceneTexture: new VariableMeta("sceneTexture", VariableKind.Texture, `texture_2d<f32>`),
    // G-buffer world normals (packed *0.5+0.5; a = surface mask) from the main pass.
    normalTexture: new VariableMeta("normalTexture", VariableKind.Texture, `texture_2d<f32>`),
    // Reverse-Z depth (depth32float) sampled as a depth texture (textureLoad only).
    // sampleType MUST be "depth" — depth32float is not a filterable float, so the
    // bind-group layout entry has to advertise Depth (else CreateBindGroup rejects it).
    depthTexture: new VariableMeta("depthTexture", VariableKind.Texture, `texture_depth_2d`, {
      textureSampleType: "depth",
    }),
    // The finest cascade (c0) AFTER merge (probeMerge[0]): rgb = merged radiance
    // carrying the full cascade hierarchy's far light, a = visibility.
    probeMerge: new VariableMeta("probeMerge", VariableKind.Texture, `texture_2d<f32>`),

    // ---- group 0 : uniforms ----
    // Camera-snapped grid origin (world XY); .zw unused. Must match the gather pass.
    gridOrigin: new VariableMeta("uGridOrigin", VariableKind.Uniform, `vec4<f32>`),
    // World height of the (single, ground) probe plane.
    probePlaneZ: new VariableMeta("uProbePlaneZ", VariableKind.Uniform, `f32`),
    // World units per probe at cascade 0.
    cell0: new VariableMeta("uCell0", VariableKind.Uniform, `f32`),
    // Stock omni light floor (matches overlay.shader.ts uAmbient).
    ambient: new VariableMeta("uAmbient", VariableKind.Uniform, `f32`),
    // Inverse view-projection (gl-matrix mat4, column-major) for world-pos reconstruction.
    invViewProj: new VariableMeta("uInvViewProj", VariableKind.Uniform, `mat4x4<f32>`),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const POSITION = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0),
    vec2f(1.0, 1.0),
    vec2f(-1.0, 1.0)
  );

// V grows DOWNWARD (top row = 0) — same convention as overlay.shader.ts.
const TEX_COORDS = array<vec2f, 6>(
    vec2f(0.0, 1.0),
    vec2f(1.0, 1.0),
    vec2f(1.0, 0.0),
    vec2f(0.0, 1.0),
    vec2f(1.0, 0.0),
    vec2f(0.0, 0.0)
  );

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) texCoord: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var output: VertexOutput;
  output.position = vec4f(POSITION[vertexIndex], 0.0, 1.0);
  output.texCoord = TEX_COORDS[vertexIndex];
  return output;
}

// Atlas/grid constants — MUST match worldGather.shader.ts and the probeRad atlas.
const GRID_DIM: u32 = ${GRID_DIM}u;
const DIR0_W:   u32 = ${DIR0_W}u;

const PI: f32 = 3.14159265;

// ===== Octahedral decode (doc §3.4, verbatim with the gather pass) =====
fn oct_decode(e_in: vec2<f32>) -> vec3<f32> {
  let e = e_in;
  var v = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
  if (v.z < 0.0) {
    v = vec3<f32>((1.0 - abs(v.yx)) * sign(v.xy), v.z);
  }
  return normalize(v);
}

// Inverse of probe_world_xy: continuous probe-grid coordinate of a world XY.
// probe_world_xy(ij) = uGridOrigin.xy + (ij + 0.5 - GRID_DIM/2) * uCell0
//   => ij = (wxy - uGridOrigin.xy) / uCell0 + GRID_DIM/2 - 0.5
fn probe_coord(wxy: vec2<f32>) -> vec2<f32> {
  return (wxy - uGridOrigin.xy) / uCell0 + f32(GRID_DIM) * 0.5 - 0.5;
}

// Integrate one probe's octahedral tile against the surface normal: sum the tile's
// radiance weighted by Lambert max(0, dot(N, dir)), normalized by per-cell solid
// angle (4*PI / dir_count). This IS the physically-correct diffuse irradiance for
// normal N — no extra "directional" hack on top (that double-counts directionality
// and shaded emitters like a ball). The probe (i,j) tile origin is (i*DIR0_W, j*DIR0_W).
fn integrate_probe(ij: vec2<i32>, N: vec3<f32>) -> vec3<f32> {
  let base = ij * i32(DIR0_W);
  var acc = vec3<f32>(0.0);
  for (var v: u32 = 0u; v < DIR0_W; v = v + 1u) {
    for (var u: u32 = 0u; u < DIR0_W; u = u + 1u) {
      let oct = ((vec2<f32>(f32(u), f32(v)) + 0.5) / f32(DIR0_W)) * 2.0 - 1.0;
      let dir = oct_decode(oct);
      let cosw = max(0.0, dot(N, dir));
      let cell = vec2<i32>(base.x + i32(u), base.y + i32(v));
      acc = acc + textureLoad(probeMerge, cell, 0).rgb * cosw;
    }
  }
  // Solid-angle normalization over the full sphere (doc §7).
  return acc * (4.0 * PI / f32(DIR0_W * DIR0_W));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let scene = textureSample(sceneTexture, textureSampler, input.texCoord);

  let px = vec2<i32>(floor(input.position.xy));
  let n = textureLoad(normalTexture, px, 0);
  if (n.a < 0.5) {
    // No surface (background): stock omni floor, no probe light.
    return vec4f(scene.rgb * uAmbient, scene.a);
  }

  // --- Reconstruct world position from reverse-Z depth (doc §7) ---
  // WebGPU clip Y is UP; texCoord V grows DOWN, so NDC.y = (1 - v)*2 - 1.
  // depth is the stored reverse-Z value (NEAR->1, FAR->0); feed it straight as NDC z.
  let depth = textureLoad(depthTexture, px, 0);
  let ndc = vec4<f32>(input.texCoord.x * 2.0 - 1.0, (1.0 - input.texCoord.y) * 2.0 - 1.0, depth, 1.0);
  let wp4 = uInvViewProj * ndc;
  let world = wp4.xyz / wp4.w;

  let N = normalize(n.rgb * 2.0 - 1.0);

  // --- Bilinearly pick the 4 probes around world.xy in the single atlas ---
  let gc = probe_coord(world.xy);
  let g0 = floor(gc);
  let f = gc - g0;                         // fractional position in the probe cell
  let i0 = vec2<i32>(g0);
  let maxIdx = i32(GRID_DIM) - 1;

  // Bilinear weights (00, 10, 01, 11).
  let w00 = (1.0 - f.x) * (1.0 - f.y);
  let w10 = f.x * (1.0 - f.y);
  let w01 = (1.0 - f.x) * f.y;
  let w11 = f.x * f.y;

  let ij00 = clamp(i0 + vec2<i32>(0, 0), vec2<i32>(0), vec2<i32>(maxIdx));
  let ij10 = clamp(i0 + vec2<i32>(1, 0), vec2<i32>(0), vec2<i32>(maxIdx));
  let ij01 = clamp(i0 + vec2<i32>(0, 1), vec2<i32>(0), vec2<i32>(maxIdx));
  let ij11 = clamp(i0 + vec2<i32>(1, 1), vec2<i32>(0), vec2<i32>(maxIdx));

  let radiance = integrate_probe(ij00, N) * w00
               + integrate_probe(ij10, N) * w10
               + integrate_probe(ij01, N) * w01
               + integrate_probe(ij11, N) * w11;
  // (weights already sum to 1; no extra normalization needed.)

  // Diffuse GI: albedo * (ambient floor + cosine-integrated incoming radiance).
  let lit = scene.rgb * (uAmbient + radiance);
  return vec4f(lit, scene.a);
}
    `,
);
