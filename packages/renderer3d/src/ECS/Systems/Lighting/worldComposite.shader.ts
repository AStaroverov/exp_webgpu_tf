import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { WORLD_DIR0_W } from "./worldGather.shader.ts";

// Stage-1 world-space Radiance Cascades — DIRECT COMPOSITE pass (doc §7).
//
// One cascade (c0), no merge. Fullscreen draw over the canvas-sized worldLitTexture.
// For each screen pixel we:
//   1. Reconstruct world position from the reverse-Z depth buffer + the inverse
//      view-projection (matching ResizeSystem.viewProjMatrix exactly, the same way
//      overlay.shader.ts projects with the forward matrix — here we go the other way).
//   2. Read the G-buffer normal (packed *0.5+0.5, a = surface mask). a < 0.5 means
//      no surface (background) -> passthrough the scene * ambient (no probe light).
//   3. Bilinearly pick the 4 probes of cascade-0 around world.xy in the merged probe
//      atlas (probeMerge, a 2D-ARRAY of uGridZ height layers). Each probe owns a
//      DIR0_W x DIR0_W octahedral tile. This is done per height layer.
//   4. Integrate the tile directions weighted by max(0, dot(N, dir)) (Lambert), with
//      solid-angle normalization 4*PI / (DIR0_W*DIR0_W), blend the 4 probes bilinearly.
//      Then blend the two layers bracketing the receiver's world height trilinearly-in-z.
//   5. lit = albedo * (ambient + radiance). The directional bonus stays consistent
//      with overlay.shader.ts (an additive, never-darkening dirGain term).
//
// COORDINATES: world is Z-up, footprints in XY. probeMerge probe (i,j,k) lives at world
// XY probe_world_xy(ij) = uGridOrigin.xy + (ij + 0.5 - (uGridX,uGridY)/2) * uCell0 on
// layer k at height baseZ + k*cellZ — IDENTICAL to the gather pass, so both agree on
// the same grid. The receiver's height is resolved by trilinear-z over k0/k1.

// Octahedral tile side — MUST match worldGather.shader.ts and the probeRad texture.
// gridX/gridY/gridZ are now passed as uniforms (height-layers Model A): the xy probe
// counts come from uGridX/uGridY and the number of z layers from uGridZ.
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
    // carrying the full cascade hierarchy's far light, a = visibility. Now a 2D-ARRAY
    // texture with one array layer per height layer (uGridZ layers); the composite
    // trilinearly blends the two layers bracketing the receiver's world height.
    probeMerge: new VariableMeta("probeMerge", VariableKind.Texture, `texture_2d_array<f32>`, {
      viewDimension: "2d-array",
    }),

    // ---- group 0 : uniforms ----
    // Camera-snapped grid origin (world XY); .zw unused. Must match the gather pass.
    gridOrigin: new VariableMeta("uGridOrigin", VariableKind.Uniform, `vec4<f32>`),
    // Probe count per side at cascade 0 (x and y can now differ).
    gridX: new VariableMeta("uGridX", VariableKind.Uniform, `u32`),
    gridY: new VariableMeta("uGridY", VariableKind.Uniform, `u32`),
    // Number of stacked horizontal probe layers (array-layer count of probeMerge).
    gridZ: new VariableMeta("uGridZ", VariableKind.Uniform, `u32`),
    // World height of layer 0, and the world-units gap between successive layers.
    baseZ: new VariableMeta("uBaseZ", VariableKind.Uniform, `f32`),
    cellZ: new VariableMeta("uCellZ", VariableKind.Uniform, `f32`),
    // World units per probe at cascade 0 (xy).
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

// Octahedral tile side — MUST match worldGather.shader.ts and the probeRad atlas.
// xy probe counts are uniforms (uGridX/uGridY); z layer count is uGridZ.
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
// probe_world_xy(ij) = uGridOrigin.xy + (ij + 0.5 - (uGridX,uGridY)/2) * uCell0
//   => ij = (wxy - uGridOrigin.xy) / uCell0 + (uGridX,uGridY)/2 - 0.5
fn probe_coord(wxy: vec2<f32>) -> vec2<f32> {
  let half = vec2<f32>(f32(uGridX), f32(uGridY)) * 0.5;
  return (wxy - uGridOrigin.xy) / uCell0 + half - 0.5;
}

// Integrate one probe's octahedral tile against the surface normal: sum the tile's
// radiance weighted by Lambert max(0, dot(N, dir)), normalized by per-cell solid
// angle (4*PI / dir_count). This IS the physically-correct diffuse irradiance for
// normal N — no extra "directional" hack on top (that double-counts directionality
// and shaded emitters like a ball). The probe (i,j) tile origin is (i*DIR0_W, j*DIR0_W).
// k selects the height-layer array slice of probeMerge.
fn integrate_probe(ij: vec2<i32>, N: vec3<f32>, k: i32) -> vec3<f32> {
  let base = ij * i32(DIR0_W);
  var acc = vec3<f32>(0.0);
  for (var v: u32 = 0u; v < DIR0_W; v = v + 1u) {
    for (var u: u32 = 0u; u < DIR0_W; u = u + 1u) {
      let oct = ((vec2<f32>(f32(u), f32(v)) + 0.5) / f32(DIR0_W)) * 2.0 - 1.0;
      let dir = oct_decode(oct);
      let cosw = max(0.0, dot(N, dir));
      let cell = vec2<i32>(base.x + i32(u), base.y + i32(v));
      acc = acc + textureLoad(probeMerge, cell, k, 0).rgb * cosw;
    }
  }
  // Solid-angle normalization over the full sphere (doc §7).
  return acc * (4.0 * PI / f32(DIR0_W * DIR0_W));
}

// One height layer's irradiance at world XY: bilinearly blend the 4 probes of layer k
// around world.xy. Factored out of fs_main so the trilinear-z blend can call it twice.
fn layerRadiance(wxy: vec2<f32>, N: vec3<f32>, k: i32) -> vec3<f32> {
  let gc = probe_coord(wxy);
  let g0 = floor(gc);
  let f = gc - g0;                         // fractional position in the probe cell
  let i0 = vec2<i32>(g0);
  let maxIdx = vec2<i32>(i32(uGridX) - 1, i32(uGridY) - 1);

  // Bilinear weights (00, 10, 01, 11).
  let w00 = (1.0 - f.x) * (1.0 - f.y);
  let w10 = f.x * (1.0 - f.y);
  let w01 = (1.0 - f.x) * f.y;
  let w11 = f.x * f.y;

  let ij00 = clamp(i0 + vec2<i32>(0, 0), vec2<i32>(0), maxIdx);
  let ij10 = clamp(i0 + vec2<i32>(1, 0), vec2<i32>(0), maxIdx);
  let ij01 = clamp(i0 + vec2<i32>(0, 1), vec2<i32>(0), maxIdx);
  let ij11 = clamp(i0 + vec2<i32>(1, 1), vec2<i32>(0), maxIdx);

  return integrate_probe(ij00, N, k) * w00
       + integrate_probe(ij10, N, k) * w10
       + integrate_probe(ij01, N, k) * w01
       + integrate_probe(ij11, N, k) * w11;
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

  // --- Trilinear-in-z over the two height layers bracketing the receiver ---
  // Layer k sits at world height baseZ + k*cellZ. lz maps the receiver's world.z
  // into continuous layer space; k0/k1 are the bracketing layers (clamped to the
  // stack), fz the blend factor. Each layer's xy irradiance is a 4-probe bilinear.
  // gridZ == 1 -> k1 == k0 and fz is harmless (mix collapses to one layer).
  let lz = (world.z - uBaseZ) / uCellZ;
  let kMax = i32(uGridZ) - 1;
  let k0 = clamp(i32(floor(lz)), 0, kMax);
  let k1 = clamp(k0 + 1, 0, kMax);
  let fz = clamp(lz - f32(k0), 0.0, 1.0);

  let radiance = mix(layerRadiance(world.xy, N, k0), layerRadiance(world.xy, N, k1), fz);

  // Diffuse GI: albedo * (ambient floor + cosine-integrated incoming radiance).
  let lit = scene.rgb * (uAmbient + radiance);
  return vec4f(lit, scene.a);
}
    `,
);
