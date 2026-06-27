import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { VoxelBakedConfig } from "./voxelConfig.ts";

// VCT Layer 3 — the full DIFFUSE HEMISPHERE cone gather. A fullscreen pass over the G-buffer:
//   1. Reconstruct the per-pixel world position P (from the reverse-Z depth + invViewProj)
//      and world normal N (from the normal G-buffer; n.rgb*2-1, n.a<0.5 = no surface).
//   2. Trace N cones over the hemisphere around the normal through the voxelRadiance mip
//      pyramid, on a procedural Fibonacci (golden-angle) hemisphere with cosine weights.
//      Each cone's diameter grows with distance (diameter = 2*aperture*dist); each step
//      samples the pyramid at LOD = log2(diameter / voxelSize) — a wider cone reads a
//      coarser mip, so one filtered fetch integrates the whole cone cross-section.
//      Front-to-back "over" compositing accumulates radiance + opacity per cone, then the
//      cones are cosine-weight-averaged → the cosine-weighted AVERAGE incoming radiance.
//   3. Write the gathered radiance (scaled by giStrength) to an HDR texture.
//
// ANGULAR RESOLUTION = SHADOW SHARPNESS. The cone count + aperture together set how finely
// the hemisphere is sampled, and per-direction occupancy (cone.a) IS the shadow term: MORE
// cones + a NARROWER aperture = sharper umbra/penumbra. But a too-narrow aperture with
// too-few cones leaves angular GAPS between cones (banding) — raise the cone count whenever
// you narrow the aperture.
//
// PERF: this pass runs at a DOWNSCALED resolution (half by default → ¼ the pixels → ~4× less
// cone work; quarter → 1/16). The downscale factor is chosen on the CPU via the cone output size;
// the shader is resolution-agnostic (maps via texCoord). The composite normal-aware-upsamples
// this output back to full res — indirect light is low-frequency, so that is fine.
//
// IMPORTANCE-SAMPLED HEMISPHERE GATHER. Wide Fibonacci fill cones at a 60° aperture read coarse
// mips, so they smear a small bright emitter across the whole cone → too dim + no directional
// shadow if used alone. So we ALSO aim one narrow cone straight at each emitter (full reach to the
// source, no fade) for a bright, sharply-shadowed contribution, folded into the SAME cosine-
// weighted hemisphere integral (acc/occAcc/wsum). The emitters are AUTO-DISCOVERED from the
// LightEmitter component every frame (no manual light list) — every emitter is treated the same.
//
// This is what produces real color bleeding. The earlier single-cone-along-the-normal form
// was the Layer-2 INTERMEDIATE (a bent-normal / AO preview); the hemisphere gather here is
// directly comparable to the brute-force gi reference, which is also a cosine-weighted
// hemisphere average.
//
// Fullscreen pass with `unproject` + reverse-Z NDC reconstruction. textureSampleLevel (explicit
// LOD) is used in the trace loop — legal in non-uniform control flow (unlike textureSample).

export function createConeShaderMeta(cfg: VoxelBakedConfig) {
  return new ShaderMeta(
  {
    // .x = screen width (px), .y = screen height (px), .z = SPARE (emitter distance-falloff is now
    // baked as the EMITTER_FALLOFF const), .w = active light count (0..8).
    params2: new VariableMeta("uParams2", VariableKind.Uniform, `vec4<f32>`),
    // Emitters to importance-sample: .xyz = world CENTER, .w = radius (penumbra source). These
    // are AUTO-DISCOVERED from the LightEmitter component (every emitter, no manual list); only
    // the first i32(uParams2.w) entries are live.
    lights: new VariableMeta("lights", VariableKind.Uniform, `array<vec4<f32>, 8>`),
    // Parallel to lights[]: .rgb = emitter color, .w = intensity → Lj = rgb·|w| is the emitter's
    // true radiance. Used to compute analytic direct light (so a blocked cone DARKENS instead of
    // picking up the occluder's own emission = the "white shadow" bug), with a bleed term so a
    // BRIGHT occluder cancels its own false shadow.
    lightColor: new VariableMeta("lightColor", VariableKind.Uniform, `array<vec4<f32>, 8>`),
    // inverse(viewProjMatrix) (reverse-Z), column-major, for world-position reconstruction.
    invViewProj: new VariableMeta("uInvViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    // .xyz = world min corner, .w = cellSize.
    gridOrigin: new VariableMeta("uGridOrigin", VariableKind.Uniform, `vec4<f32>`),
    // .xyz = voxel counts per axis.
    gridDims: new VariableMeta("uGridDims", VariableKind.Uniform, `vec4<i32>`),
    // G-buffer reverse-Z depth (texture_depth_2d) + world normal (rgba16float, packed *0.5+0.5).
    depthTex: new VariableMeta("depthTex", VariableKind.Texture, `texture_depth_2d`, {
      textureSampleType: "depth",
    }),
    normalTex: new VariableMeta("normalTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    // The voxelRadiance mip pyramid (ALL mips) — the cone reads it at the per-step LOD (for the
    // aimed emitter cones + the short AO cones).
    voxelRadiance: new VariableMeta("voxelRadiance", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    // Irradiance-probe SH-L1 volume (one texture per color channel, .xyzw = the 4 SH coeffs).
    // Sampled trilinearly (voxelSampler) to reconstruct the low-frequency bounce that REPLACES the
    // old per-pixel fill hemisphere cones.
    shR: new VariableMeta("shR", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    shG: new VariableMeta("shG", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    shB: new VariableMeta("shB", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    // Filtering sampler for textureSampleLevel over the pyramid + the SH volume.
    voxelSampler: new VariableMeta("voxelSampler", VariableKind.Sampler, `sampler`),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
// BAKED tuning consts (interpolated from VoxelBakedConfig at shader-build time). See voxelConfig.ts.
const NORMAL_BIAS: f32 = ${cfg.normalBias};
const APERTURE: f32 = ${cfg.aperture};
const GI_STRENGTH: f32 = ${cfg.giStrength};
const EMITTER_FALLOFF: f32 = ${cfg.emitterFalloff};
const EMITTER_DIRECT: f32 = ${cfg.emitterDirect};
const AIMED_STEPS: i32 = ${cfg.aimedSteps};
const AIMED_ALPHA_CUT: f32 = ${cfg.aimedAlphaCut};
const AO_CONE_COUNT: i32 = ${cfg.aoConeCount};
const AO_REACH: f32 = ${cfg.aoReach};
const AO_STEPS: i32 = ${cfg.aoSteps};

const POSITION = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);
const TEX_COORDS = array<vec2f, 6>(
  vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0),
  vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(0.0, 0.0)
);

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) texCoord: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4f(POSITION[vertexIndex], 0.0, 1.0);
  out.texCoord = TEX_COORDS[vertexIndex];
  return out;
}

// Unproject an NDC point (z reverse-Z) to world space.
fn unproject(ndc: vec3<f32>) -> vec3<f32> {
  let w = uInvViewProj * vec4<f32>(ndc, 1.0);
  return w.xyz / w.w;
}

// Orthonormal basis with column 2 = n (so basis * (x,y,z) = x*t + y*b + z*n).
fn build_basis(n: vec3<f32>) -> mat3x3<f32> {
  let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
  let t = normalize(cross(a, n));
  let b = cross(n, t);
  return mat3x3<f32>(t, b, n);
}

// One cone marched along dir from origin: diameter grows with distance, each step samples
// voxelRadiance at LOD=log2(diameter/voxelSize) and composites front-to-back ("over").
// reach = how far this cone marches (world units). The step is floored at reach/maxSteps so the
// cone ALWAYS spans its full reach within the step budget — without this, a narrow cone's
// tiny near-field steps burn the budget and it dies ~16 units out (the "light stops at cone
// reach" problem). fadeFrac>0 tapers the gathered radiance over the last fadeFrac of reach
// (hides the artificial cutoff of the fill cones); pass 0 for aimed cones, which end at the
// real light, so they must NOT be tapered. Occlusion (alpha) is never tapered → shadows stay.
// maxSteps = march budget (aimed cones want it high for sharp shadows; the low-frequency fill
// hemisphere is fine with ~half — its bounce is smooth, so fewer steps barely change it but
// halve the dominant cone cost). alphaCut = early-out opacity: a fill cone that is ~opaque adds
// almost nothing more, so cutting at <1 saves the tail steps; aimed cones pass 1.0 (no early cut).
fn trace_cone(origin: vec3<f32>, dir: vec3<f32>, aperture: f32, reach: f32, fadeFrac: f32, startJ: f32, maxSteps: i32, alphaCut: f32) -> vec4<f32> {
  var col = vec3<f32>(0.0);
  var alpha = 0.0;
  let voxelSize = uGridOrigin.w;
  let gridMin = uGridOrigin.xyz;
  let extent = vec3<f32>(uGridDims.xyz) * voxelSize;
  let stepFloor = reach / f32(maxSteps);
  // Per-pixel SCALE of the start distance (startJ in [0,1)). Because the march is ~geometric
  // (step grows with distance), scaling dist0 shifts the sampling "shells" by a per-pixel
  // factor at EVERY distance → the concentric "tree-ring" banding around bright sources
  // dithers across pixels and is blurred away by the half-res upsample. Lets few cones look
  // smooth without going to 48.
  var dist = voxelSize * (1.0 + startJ);
  for (var i = 0; i < maxSteps; i = i + 1) {
    if (alpha >= alphaCut || dist > reach) { break; }
    let diameter = max(voxelSize, 2.0 * aperture * dist);
    let lod = log2(diameter / voxelSize);
    let wp = origin + dir * dist;
    let uvw = (wp - gridMin) / extent;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) { break; }
    let s = textureSampleLevel(voxelRadiance, voxelSampler, uvw, lod);
    var window = 1.0;
    if (fadeFrac > 0.0) {
      window = clamp((reach - dist) / (reach * fadeFrac), 0.0, 1.0);
    }
    col = col + (1.0 - alpha) * s.rgb * window;
    alpha = alpha + (1.0 - alpha) * s.a;
    dist = dist + max(diameter * 0.5, stepFloor);
  }
  return vec4<f32>(col, alpha);
}

// Reconstruct the cosine-weighted AVERAGE radiance from an SH-L1 channel (.xyzw = L00,L1m1,L10,L11)
// for surface normal N. The SH-L1 diffuse IRRADIANCE is E = 0.886227*L00 + 1.023328*(L1·N); we
// divide by PI to get the average incoming radiance (for uniform radiance L, E/PI = L), so this
// matches the scale of the OLD per-cone fill average it replaces. Clamped against SH ringing.
fn sh_avg_radiance(c: vec4<f32>, N: vec3<f32>) -> f32 {
  let E = 0.886227 * c.x + 1.023328 * (c.y * N.y + c.z * N.z + c.w * N.x);
  return max(0.0, E * 0.31830989); // * (1/PI)
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  // This pass renders at a downscaled res (half or quarter — set on the CPU by the cone output
  // size). Map this cone pixel to a representative full-res G-buffer texel via the normalized
  // texCoord, which spans [0,1] across the target at ANY resolution → no scale uniform needed.
  // uParams2.xy carries the FULL canvas dims. (half below is still the per-pixel jitter seed.)
  let half = vec2<i32>(floor(input.position.xy));
  let full = min(vec2<i32>(input.texCoord * uParams2.xy), vec2<i32>(uParams2.xy) - vec2<i32>(1));

  // World normal from the G-buffer; a<0.5 = no surface at this pixel.
  let n = textureLoad(normalTex, full, 0);
  if (n.a < 0.5) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  let N = normalize(n.rgb * 2.0 - 1.0);

  // Reconstruct world position from reverse-Z depth.
  let depth = textureLoad(depthTex, full, 0);
  let uv = (vec2<f32>(full) + vec2<f32>(0.5)) / uParams2.xy;
  let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth);
  let P = unproject(ndc);

  // Lift the cone origin off the surface to avoid self-sampling the originating voxel.
  let cellSize = uGridOrigin.w;
  let origin = P + N * (cellSize * 1.5 + NORMAL_BIAS);

  // Hemisphere basis (for the AO cones below). The fill hemisphere now comes from the probe SH
  // volume; the per-pixel cone count is gone (replaced by the baked AO_CONE_COUNT for contact AO).
  let basis = build_basis(N);
  let aperture = APERTURE;
  // Per-pixel azimuth rotation of the whole cone set (interleaved-gradient-noise hash) so the
  // gaps between narrow cones fall in DIFFERENT directions each pixel → fixed "ray" banding
  // becomes fine dither, which the half-res + bilinear upsample then blurs away. Lets a narrow
  // aperture use FEWER cones without visible streaks.
  let jitter = fract(52.9829189 * fract(dot(vec2<f32>(half), vec2<f32>(0.06711056, 0.00583715)))) * 6.2831853;
  // Decorrelated [0,1) per-pixel value for radial (start-distance) dither — kills the
  // concentric "tree-ring" banding around bright sources at low cone counts.
  let jrad = fract(jitter * 1.61803399);

  // Emitter DIRECT light, SUMMED on the SAME scale as the sun (NO /wsum averaging) so a brighter
  // emitter actually competes with (and overpowers) the sun — e.g. a bright lamp near the floor
  // fills the sun-shadow it casts on itself. occAcc = AO occlusion from the short AO cones below.
  var directEmitters = vec3<f32>(0.0);
  var occAcc = 0.0;

  // (a) AIMED cones — one narrow cone per emitter, pointed straight at the light center for a
  // bright, far-reaching direct + soft-shadow term. Aperture = the light's angular size
  // (radius / distance) sets the penumbra width; the cosine term (ndl) is its weight, so these
  // fold into the SAME hemisphere integral as the fill cones below. Emitters are auto-discovered
  // (uParams2.w = live count, lights[] = world centers + radius).
  let lc = min(8, max(0, i32(uParams2.w)));
  for (var j = 0; j < lc; j = j + 1) {
    let toL = lights[j].xyz - origin;
    let d = length(toL);
    if (d < 1e-3) { continue; }
    let dir = toL / d;
    let ndl = max(dot(N, dir), 0.0);
    if (ndl <= 0.0) { continue; }
    // Distance falloff: without it the direct light is a FLAT-bright pool with a hard rim (reads as
    // an "invisible bigger sphere" the surface cuts through). atten = 1 at the light center, ~1/d²
    // far. EMITTER_FALLOFF = falloff coefficient (0 = none → flat sun-like emitter; 1 = standard);
    // lr = emitter radius (lights[j].w) so it is 1 inside the source and fades smoothly outside.
    let lr = max(lights[j].w, 1e-3);
    let atten = 1.0 / (1.0 + EMITTER_FALLOFF * (d * d) / (lr * lr));
    let full = ndl * lightColor[j].rgb * abs(lightColor[j].w) * atten;
    // LIGHT CULL: the 64-step shadow march is the dominant cost. Compute this emitter's direct
    // contribution FIRST (cheap, no trace) and skip the march entirely if it is negligible here —
    // far away, steep grazing, dim, or faded out by the falloff. For "many small emitters" most
    // pixels see only a few relevant lights, so this is a big win at ~zero visible change (the
    // skipped term was ≈0). With falloff=0 (flat sun-like) atten stays 1, so global lights are
    // never culled — exactly the intended behavior for that mode.
    if (max(full.r, max(full.g, full.b)) * EMITTER_DIRECT < 0.003) { continue; }
    let ap = clamp(lights[j].w / d, 0.02, 0.5);   // angular size of the light = penumbra width
    // AIMED: step budget + early-out opacity baked (AIMED_STEPS, AIMED_ALPHA_CUT). Defaults (32, 1.0)
    // = crisp behavior; lower steps / a <1 alphaCut trade shadow precision for speed on heavy scenes.
    let r = trace_cone(origin, dir, ap, d, 0.0, jrad, max(1, AIMED_STEPS), AIMED_ALPHA_CUT);
    // ANALYTIC DIRECT + bleed-cancel (fixes the "white shadow"). full = the emitter's own light.
    // shadow = how much the cone's opacity removes; bleed = the radiance the cone actually
    // gathered along the way (a BRIGHT occluder => big bleed => its false shadow is cancelled;
    // a DARK occluder => ~0 bleed => the shadow survives). The target emitter itself bleeds ≈ its
    // own Lj, so it always delivers full light. Clamped so it can only darken, never exceed full.
    let occ = clamp(r.a, 0.0, 1.0);
    let shadow = full * occ;
    let bleed = ndl * r.rgb;
    let contrib = full - max(vec3<f32>(0.0), shadow - bleed);
    directEmitters = directEmitters + max(vec3<f32>(0.0), contrib);
  }

  // (b) FILL / bounce — the irradiance-probe SH volume (built once per frame in voxelProbe). One
  // trilinear SH fetch per channel + a cosine-weighted reconstruction = the low-frequency
  // hemisphere bounce the old per-pixel fill cones produced, at O(1) cost. Added as its OWN term
  // (scaled by giStrength below) — no longer averaged against the emitter direct.
  let pUvw = (origin - uGridOrigin.xyz) / (vec3<f32>(uGridDims.xyz) * cellSize);
  let cR = textureSampleLevel(shR, voxelSampler, pUvw, 0.0);
  let cG = textureSampleLevel(shG, voxelSampler, pUvw, 0.0);
  let cB = textureSampleLevel(shB, voxelSampler, pUvw, 0.0);
  let fillAvg = vec3<f32>(sh_avg_radiance(cR, N), sh_avg_radiance(cG, N), sh_avg_radiance(cB, N));

  // (c) AO — a few SHORT hemisphere occlusion cones (opacity ONLY, no radiance → no double-count
  // with the probe bounce). Becomes the .a/visibility output the composite reads as ambient
  // occlusion; kept per-pixel + short so small moving objects still darken their contacts.
  let aoCount = AO_CONE_COUNT;
  let aoReach = AO_REACH;
  let aoSteps = max(1, AO_STEPS);
  var aoW = 0.0;
  for (var a = 0; a < aoCount; a = a + 1) {
    let k = (f32(a) + 0.5) / f32(aoCount);
    let cosT = 1.0 - k;
    let sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    let phi = f32(a) * 2.39996323 + jitter;
    let local = vec3<f32>(sinT * cos(phi), sinT * sin(phi), cosT);
    let dir = normalize(basis * local);
    let r = trace_cone(origin, dir, aperture, aoReach, 0.0, jrad, aoSteps, 0.95);
    occAcc = occAcc + cosT * r.a;
    aoW = aoW + cosT;
  }

  // Combine: emitter DIRECT (summed, sun-scale) × EMITTER_DIRECT strength + probe bounce ×
  // GI_STRENGTH. No /wsum averaging → a bright emitter competes with the sun.
  let indirect = directEmitters * EMITTER_DIRECT + fillAvg * GI_STRENGTH;
  let visibility = clamp(1.0 - occAcc / max(aoW, 1e-4), 0.0, 1.0);
  // rgb = emitter direct + indirect bounce; a = AO visibility, read by the composite.
  return vec4f(indirect, visibility);
}
`,
  );
}
