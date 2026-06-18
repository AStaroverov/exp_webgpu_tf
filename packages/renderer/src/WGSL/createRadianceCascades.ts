import { GPUShader } from "./GPUShader.ts";
import { ShaderMeta } from "./ShaderMeta.ts";
import { VariableKind, VariableMeta } from "../Struct/VariableMeta.ts";
import { wgsl } from "./wgsl.ts";
import {
  cameraPosition,
  invViewProjection,
  projectionMatrix,
} from "../ECS/Systems/ResizeSystem.ts";

/**
 * Phase 2 lighting — SCREEN-SPACE RADIANCE CASCADES over the G-buffer.
 *
 * Replaces the stopgap Lambert composite. Because Phase 1 already produced a
 * reverse-Z depth buffer, visibility is computed by MARCHING THE DEPTH BUFFER in
 * screen space (no JFA / distance field): a probe reconstructs its world position
 * from depth + invViewProjection, casts screen-space-2D direction rays, and at each
 * step re-projects the world sample to screen and compares its reverse-Z ndc.z to the
 * stored depth. A hit samples gEmission (+ a cheap albedo fill) as the seeded radiance;
 * a miss falls through to the upper cascade (merge) or the sky/ambient term.
 *
 * Pass graph (all fullscreen draw(6) render passes, rcW x rcH unless noted):
 *   1. cascade/merge loop  : n = CASCADE_COUNT-1 .. 0, fused raycast+merge,
 *                            ping-pong cascA <-> cascB. Top cascade has no upper input.
 *   2. gather (final)      : cascade0 -> per-pixel irradiance, writes bgra8unorm (full).
 *
 * Reverse-Z reminder (pinned everywhere): depthClearValue 0, near -> 1, far -> 0.
 * "stored surface is IN FRONT of my sample" <=> storedDepth > sampleNdcZ.
 *
 * KNOWN Phase-2 limitation: screen-space only — off-screen / behind-surface
 * occluders are invisible (single depth layer), so expect light leaking at
 * silhouettes; the ambient/sky fallback hides most of it. World-space probe RC is
 * the documented later upgrade.
 */

// ============================================================================
// CASCADE STRUCTURE (see docs/3D_MIGRATION.md Phase 2 + the RC spec)
// ============================================================================
export const CASCADE_COUNT = 6; // fixed; static loop/uniform, plenty for a ~400px RC buffer
const BASE_RAYS = 16; // rays per probe at c0; quadruples per cascade (RC angular invariant).
// 16 (vs 4) kills the angular banding/faceting in the lit irradiance. FREE in
// fragment cost (same texture, 1 texel = 1 ray); only trades spatial probe density.
// INVARIANT: PROBE0_SPACING_TX must equal raysPerSide(0) = sqrt(BASE_RAYS) so probe
// cells tile the cascade texture exactly (probeCenterTexel assumes spacing == rps).
const BASE_INTERVAL_PX = 8.0; // c0 ray length in full-res screen px; doubles per cascade
const PROBE0_SPACING_TX = 4.0; // = sqrt(BASE_RAYS); c0 probe every 4 RC-texels; doubles per cascade
const MARCH_STEPS = 8; // per-ray screen-space march budget (default; tune in GUI)
const ZBIAS_WORLD = 2.0; // occlusion dead-zone in WORLD units (kills self-occlusion acne)
const AMBIENT = 0.18; // diffuse ambient floor (final gather)
const AMBIENT_FILL = 0.25; // cheap albedo contribution added at a hit
// Sky / ambient color a missed ray carries — warm fill matching the bg.
const SKY = [0.18, 0.15, 0.11] as const;
// Edge-aware À-TROUS denoise of the (noisy, few-ray) irradiance. Each iteration is
// a 5x5 cross-bilateral (B3-spline weights) whose tap STRIDE doubles (1,2,4,8,...),
// so N iterations cover an effective radius ~2^N px at N*25 taps instead of (2^N)^2.
// Edge-stop weights (world-position + normal) keep light from bleeding across
// silhouettes — the wide reach is what reconstructs the coarse RC probe grid.
const MAX_ATROUS = 6; // hard cap (per-iteration uniform buffers are preallocated)
const DENOISE_ITERS = 5; // à-trous iterations (0 = off); effective radius ~2^ITERS px
const DENOISE_WORLD_SIGMA = 14; // base world-distance edge stop (units), scaled by stride/pass
const DENOISE_NORMAL_POW = 32; // normal edge-stop sharpness (higher = stricter)

// Per-cascade derived constants, precomputed on CPU (uploaded each pass).
function rayCount(n: number) {
  return BASE_RAYS * 4 ** n;
}
function raysPerSide(n: number) {
  return Math.round(Math.sqrt(rayCount(n)));
}
function probeSpacing(n: number) {
  return PROBE0_SPACING_TX * 2 ** n;
}

// ============================================================================
// Shared WGSL — direction-first tiling helpers + depth march. Pinned ONCE so the
// cascade pass, the merge fetchUpper, and the final gather all agree on the layout.
// ============================================================================
// language=WGSL
const RC_COMMON = /* wgsl */ `
const TWO_PI: f32 = 6.28318530718;
const SKY_DEPTH_EPS: f32 = 1e-6; // sampled depth <= this => far plane / background

// Fullscreen triangle-pair. uv.y==0 is the TOP row (matches createPresent TEX).
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

// uRC packing (array<vec4f,5> = 80B; getTypeBufferSize parses it):
//   [0].xy = rcSize (px),            [0].zw = 1/rcSize
//   [1].x  = cascadeIndex (f32),     [1].y  = rayCount
//   [1].z  = raysPerSide,            [1].w  = probeSpacing (this cascade)
//   [2].x  = probeSpacingUpper,      [2].y  = tNear (full-res px)
//   [2].z  = tFar (full-res px),     [2].w  = isTop (1.0 at top cascade)
//   [3].x  = near, [3].y = far,      [3].z  = zBias, [3].w = ambientFill
//   [4].x  = marchSteps,             [4].yzw = sky color (rgb)
fn rcSize()      -> vec2f { return uRC[0].xy; }
fn rcInv()       -> vec2f { return uRC[0].zw; }
fn rayCountF()   -> f32   { return uRC[1].y; }
fn raysPerSideF()-> f32   { return uRC[1].z; }
fn probeSpacing()-> f32   { return uRC[1].w; }
fn probeSpacingUpper() -> f32 { return uRC[2].x; }
fn tNearPx()     -> f32   { return uRC[2].y; }
fn tFarPx()      -> f32   { return uRC[2].z; }
fn isTop()       -> f32   { return uRC[2].w; }
fn nearZ()       -> f32   { return uRC[3].x; }
fn farZ()        -> f32   { return uRC[3].y; }
fn zBias()       -> f32   { return uRC[3].z; }
fn ambientFill() -> f32   { return uRC[3].w; }
fn marchSteps()  -> f32   { return uRC[4].x; }
fn skyColor()    -> vec3f { return uRC[4].yzw; }

// ---- reverse-Z depth load (nearest; depth32float is unfilterable) ----------
fn loadDepth(px: vec2i) -> f32 {
  return textureLoad(depthTex, clamp(px, vec2i(0), vec2i(uScreen.xy) - vec2i(1)), 0).r;
}
fn loadDepthUV(uv: vec2f) -> f32 {
  return loadDepth(vec2i(uv * uScreen.xy));
}

// ---- world reconstruction from screen uv + reverse-Z depth -----------------
// uv has v DOWN (uv.y==0 = top). NDC is y-UP, flip v. depth is already reverse-Z
// ndc.z; uInvViewProj is the inverse of that SAME reverse-Z VP -> no remap.
fn worldFromDepth(uv: vec2f, ndcZ: f32) -> vec3f {
  let ndc = vec4f(uv.x * 2. - 1., 1. - uv.y * 2., ndcZ, 1.);
  let w = uInvViewProj * ndc;
  return w.xyz / w.w;
}

// ---- project a world point back to screen uv + its reverse-Z ndc.z ---------
struct Proj { uv: vec2f, ndcZ: f32, behind: bool };
fn projectToScreen(world: vec3f) -> Proj {
  let clip = uViewProj * vec4f(world, 1.);
  var o: Proj;
  o.behind = clip.w <= 0.;
  let invW = 1. / clip.w;
  let ndc = clip.xyz * invW;
  o.uv = vec2f(ndc.x * .5 + .5, .5 - ndc.y * .5); // ndc-up -> screen-down (inverse of above)
  o.ndcZ = ndc.z;
  return o;
}

// ============================================================================
// Direction-first tiling. A cascade texture is probeCount x probeCount CELLS;
// each cell is raysPerSide x raysPerSide texels (one per direction).
// ============================================================================
struct TexelAddr { cell: vec2i, dirIndex: i32 };
fn decodeTexel(px: vec2i, rps: i32) -> TexelAddr {
  var a: TexelAddr;
  a.cell = px / rps;
  let d = px % rps;
  a.dirIndex = d.y * rps + d.x;
  return a;
}
// RC-texel center of a probe cell, in RC-texel coords.
fn probeCenterTexel(cell: vec2i, spacing: f32) -> vec2f {
  return (vec2f(cell) + 0.5) * spacing;
}
// 2D screen-space fan direction for a direction slot.
fn dirForIndex(dirIndex: i32, rc: f32) -> vec2f {
  let angle = (f32(dirIndex) + 0.5) * (TWO_PI / rc);
  return vec2f(cos(angle), sin(angle));
}

// ---- THE DEPTH MARCH (screen-space) ----------------------------------------
// probeUV: screen uv of the probe (its surface point). rayDir2: screen-space dir.
// Marches the [tNear,tFar] shell in full-res screen pixels. Returns rgb radiance,
// a=1 on hit/sky-terminate (resolved), a=0 if still unresolved (needs merge).
fn marchRay(probeUV: vec2f, rayDir2: vec2f) -> vec4f {
  let dNear = tNearPx();
  let dFar  = tFarPx();
  let span  = max(dFar - dNear, 0.0);
  let steps = max(marchSteps(), 1.0);
  let stepPx = span / steps;
  // step in screen UV along the 2D ray.
  let stepUV = rayDir2 * (stepPx * uScreen.zw); // zw = 1/(w,h)
  var uv = probeUV + rayDir2 * (dNear * uScreen.zw);

  // Probe surface world pos + its distance to the camera. This is the LINEAR
  // basis for the occlusion test: comparing reverse-Z ndc directly is unusable
  // (ndc is wildly non-linear over [near,far], so a constant ndc bias either does
  // nothing or skips every occluder). World distance makes zBias a real world-unit
  // dead-zone that kills self-occlusion acne without flattening real contact.
  let probeZ = loadDepthUV(probeUV);
  let probeW = worldFromDepth(probeUV, probeZ);
  let probeDist = length(probeW - uCamPos.xyz);

  let stepsI = i32(steps);
  for (var i = 0; i < stepsI; i = i + 1) {
    uv = uv + stepUV;
    if (uv.x < 0. || uv.x > 1. || uv.y < 0. || uv.y > 1.) { break; } // exited screen -> miss
    let sceneZ = loadDepthUV(uv);
    if (sceneZ <= SKY_DEPTH_EPS) { continue; } // background here, keep marching
    // The depth surface at this screen step is CLOSER to the camera than the probe
    // (by more than zBias world units) -> it occludes this screen-space ray.
    let sampleW = worldFromDepth(uv, sceneZ);
    let sampleDist = length(sampleW - uCamPos.xyz);
    if (sampleDist < probeDist - zBias()) {
      // occluder in front -> this ray is blocked; seed radiance from the occluder.
      let lit = textureSampleLevel(gEmission, samp, uv, 0.).rgb
              + textureSampleLevel(gAlbedo,   samp, uv, 0.).rgb * ambientFill();
      return vec4f(lit, 1.0);
    }
  }
  // ran out of steps / left screen with no occluder -> sky. Terminate only at the
  // top cascade so lower cascades still merge the sky from above.
  return vec4f(skyColor(), select(0.0, 1.0, isTop() > 0.5));
}
`;

// ============================================================================
// CASCADE PASS — fused raycast + merge. One texel = (cell, dirIndex) of cascade n.
// reads the UPPER cascade (n+1) via cascadeIn for the merge.
// ============================================================================
const cascadeMeta = new ShaderMeta(
  {
    samp: new VariableMeta("samp", VariableKind.Sampler, `sampler`),
    depthTex: new VariableMeta("depthTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "unfilterable-float",
    }),
    gAlbedo: new VariableMeta("gAlbedo", VariableKind.Texture, `texture_2d<f32>`),
    gEmission: new VariableMeta("gEmission", VariableKind.Texture, `texture_2d<f32>`),
    cascadeIn: new VariableMeta("cascadeIn", VariableKind.Texture, `texture_2d<f32>`),
    uInvViewProj: new VariableMeta("uInvViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    uViewProj: new VariableMeta("uViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    uCamPos: new VariableMeta("uCamPos", VariableKind.Uniform, `vec4<f32>`),
    uScreen: new VariableMeta("uScreen", VariableKind.Uniform, `vec4<f32>`),
    uRC: new VariableMeta("uRC", VariableKind.Uniform, `array<vec4<f32>, 5>`),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
${RC_COMMON}

// fetch an UPPER-cascade (n+1) texel: (parentCell, childDir) -> texel -> sample.
fn fetchUpper(parentCell: vec2f, childDir: i32, rpsUpper: i32) -> vec3f {
  let cellI = vec2i(clamp(parentCell, vec2f(0.), rcSize() - 1.));
  let d = vec2i(childDir % rpsUpper, childDir / rpsUpper);
  let texel = cellI * rpsUpper + d;
  let uv = (vec2f(texel) + 0.5) * rcInv();
  return textureSampleLevel(cascadeIn, samp, uv, 0.).rgb;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let rps = i32(raysPerSideF());
  let px = vec2i(in.uv * rcSize());
  let addr = decodeTexel(px, rps);

  // probe surface in screen uv (RC-texel center -> full-res uv via /rcSize)
  let probeTexel = probeCenterTexel(addr.cell, probeSpacing());
  let probeUV = probeTexel * rcInv();

  // skip probes that land on background (no surface to light from)
  let probeZ = loadDepthUV(probeUV);
  if (probeZ <= SKY_DEPTH_EPS) {
    return vec4f(skyColor(), 1.0);
  }

  let rayDir2 = dirForIndex(addr.dirIndex, rayCountF());
  let cur = marchRay(probeUV, rayDir2);

  // hit, or top cascade: nothing above to merge.
  if (cur.a > 0.5 || isTop() > 0.5) {
    return cur;
  }

  // ---- MERGE with cascade n+1 (4x directions, 2x coarser probe grid) ----
  let rpsUpper = rps * 2; // raysPerSide(n+1) = 2 * raysPerSide(n)
  let childBase = addr.dirIndex * 4; // dirs ordered: 4 children per parent dir
  // parent-grid coordinate of this probe cell.
  let upperProbeF = (vec2f(addr.cell) * probeSpacing()) / probeSpacingUpper() - 0.5;
  let p0 = floor(upperProbeF);
  let f = upperProbeF - p0;
  var acc = vec3f(0.0);
  for (var c = 0; c < 4; c = c + 1) {
    let cd = childBase + c;
    let s00 = fetchUpper(p0 + vec2f(0., 0.), cd, rpsUpper);
    let s10 = fetchUpper(p0 + vec2f(1., 0.), cd, rpsUpper);
    let s01 = fetchUpper(p0 + vec2f(0., 1.), cd, rpsUpper);
    let s11 = fetchUpper(p0 + vec2f(1., 1.), cd, rpsUpper);
    let sx0 = mix(s00, s10, f.x);
    let sx1 = mix(s01, s11, f.x);
    acc = acc + mix(sx0, sx1, f.y);
  }
  acc = acc * 0.25; // average the 4 angular children
  return vec4f(cur.rgb + acc, 1.0);
}
  `,
);

// ============================================================================
// GATHER PASS — cascade0 -> per-pixel irradiance, one diffuse bounce, bgra8unorm.
// ============================================================================
const gatherMeta = new ShaderMeta(
  {
    samp: new VariableMeta("samp", VariableKind.Sampler, `sampler`),
    casc0: new VariableMeta("casc0", VariableKind.Texture, `texture_2d<f32>`),
    uRCG: new VariableMeta("uRCG", VariableKind.Uniform, `vec4<f32>`),
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

struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VOut {
  var o: VOut;
  o.pos = vec4f(POSITION[i], 0., 1.);
  o.uv = TEX[i];
  return o;
}

// Irradiance of ONE c0 probe cell: average its rps0 x rps0 directional texels.
fn cellIrradiance(cell: vec2i, rps0: i32, rcSize: vec2f) -> vec3f {
  let maxCell = vec2i(rcSize / f32(rps0)) - vec2i(1);
  let cc = clamp(cell, vec2i(0), maxCell); // edge cells clamp (no wrap)
  var irr = vec3f(0.0);
  let dirs = rps0 * rps0;
  for (var d = 0; d < dirs; d = d + 1) {
    let dd = vec2i(d % rps0, d / rps0);
    let texel = cc * rps0 + dd;
    let uv = (vec2f(texel) + 0.5) / rcSize;
    irr = irr + textureSampleLevel(casc0, samp, uv, 0.).rgb;
  }
  return irr / f32(dirs);
}

// uRCG: .xy = rcSize, .z = raysPerSide(0), .w = unused
// Outputs RAW IRRADIANCE (rgba16float) — the denoise+composite pass blurs it
// (edge-aware) and applies albedo/emission/ambient. Splitting them lets the
// denoiser smooth only the noisy light, never the crisp albedo detail.
@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let rcSize = uRCG.xy;
  let rps0 = i32(uRCG.z);

  // Continuous probe-cell coordinate. Probe centers sit at (cell+0.5)*spacing, so
  // subtract 0.5 to land between cells, then BILINEARLY blend the 4 neighbours —
  // this removes the per-cell blockiness of picking a single truncated cell.
  let probeTexel = in.uv * rcSize;
  let cellF = probeTexel / ${PROBE0_SPACING_TX} - 0.5;
  let c0 = vec2i(floor(cellF));
  let f = cellF - floor(cellF);
  let s00 = cellIrradiance(c0 + vec2i(0, 0), rps0, rcSize);
  let s10 = cellIrradiance(c0 + vec2i(1, 0), rps0, rcSize);
  let s01 = cellIrradiance(c0 + vec2i(0, 1), rps0, rcSize);
  let s11 = cellIrradiance(c0 + vec2i(1, 1), rps0, rcSize);
  let irr = mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
  return vec4f(irr, 1.0);
}
  `,
);

// ============================================================================
// À-TROUS PASS — one edge-aware wavelet iteration of the irradiance. 5x5 cross-
// bilateral with B3-spline weights, tap STRIDE = uAT.x (doubles each iteration).
// Guided by world position + normal (G-buffer) so it never blurs across edges.
// Reads srcTex (irradiance), writes irradiance (ping-pong); rgba16float.
// ============================================================================
const atrousMeta = new ShaderMeta(
  {
    samp: new VariableMeta("samp", VariableKind.Sampler, `sampler`),
    srcTex: new VariableMeta("srcTex", VariableKind.Texture, `texture_2d<f32>`),
    gNormal: new VariableMeta("gNormal", VariableKind.Texture, `texture_2d<f32>`),
    depthTex: new VariableMeta("depthTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "unfilterable-float",
    }),
    uInvViewProj: new VariableMeta("uInvViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    uScreen: new VariableMeta("uScreen", VariableKind.Uniform, `vec4<f32>`),
    uAT: new VariableMeta("uAT", VariableKind.Uniform, `vec4<f32>`),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const SKY_DEPTH_EPS: f32 = 1e-6;
const POSITION = array<vec2f, 6>(
  vec2f(-1., -1.), vec2f(1., -1.), vec2f(1., 1.),
  vec2f(-1., -1.), vec2f(1., 1.), vec2f(-1., 1.));
const TEX = array<vec2f, 6>(
  vec2f(0., 1.), vec2f(1., 1.), vec2f(1., 0.),
  vec2f(0., 1.), vec2f(1., 0.), vec2f(0., 0.));
// B3-spline à-trous kernel (1,4,6,4,1)/16, applied as a separable outer product.
const KW = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);

struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VOut {
  var o: VOut;
  o.pos = vec4f(POSITION[i], 0., 1.);
  o.uv = TEX[i];
  return o;
}

fn loadDepthUV(uv: vec2f) -> f32 {
  let px = clamp(vec2i(uv * uScreen.xy), vec2i(0), vec2i(uScreen.xy) - vec2i(1));
  return textureLoad(depthTex, px, 0).r;
}
fn worldFromDepth(uv: vec2f, ndcZ: f32) -> vec3f {
  let ndc = vec4f(uv.x * 2. - 1., 1. - uv.y * 2., ndcZ, 1.);
  let w = uInvViewProj * ndc;
  return w.xyz / w.w;
}

// uAT: .x = stride (px), .y = worldSigma (already stride-scaled), .z = normalPow
@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let center = textureSampleLevel(srcTex, samp, in.uv, 0.).rgb;
  let dC = loadDepthUV(in.uv);
  if (dC <= SKY_DEPTH_EPS) { return vec4f(center, 1.0); } // background: pass through

  let stride = uAT.x;
  let ws = max(uAT.y, 1e-3);
  let npow = uAT.z;
  let nC = normalize(textureSampleLevel(gNormal, samp, in.uv, 0.).xyz);
  let pC = worldFromDepth(in.uv, dC);
  let texel = uScreen.zw;

  var sum = vec3f(0.0);
  var wsum = 0.0;
  for (var j = 0; j < 5; j = j + 1) {
    for (var i = 0; i < 5; i = i + 1) {
      let suv = in.uv + vec2f(f32(i - 2), f32(j - 2)) * stride * texel;
      let dS = loadDepthUV(suv);
      if (dS <= SKY_DEPTH_EPS) { continue; } // skip background neighbours
      let nS = normalize(textureSampleLevel(gNormal, samp, suv, 0.).xyz);
      let pS = worldFromDepth(suv, dS);
      let wn = pow(max(dot(nC, nS), 0.0), npow); // normal edge-stop
      let dd = pC - pS;
      let wd = exp(-dot(dd, dd) / (ws * ws));     // world-distance edge-stop
      let w = KW[i] * KW[j] * wn * wd;
      sum = sum + textureSampleLevel(srcTex, samp, suv, 0.).rgb * w;
      wsum = wsum + w;
    }
  }
  if (wsum > 0.0) { return vec4f(sum / wsum, 1.0); }
  return vec4f(center, 1.0);
}
  `,
);

// ============================================================================
// COMPOSITE PASS — the diffuse bounce: lit = emission + albedo*(ambient + irr).
// Reads the (denoised) irradiance + G-buffer albedo/emission. Full-res bgra8unorm.
// ============================================================================
const compositeMeta = new ShaderMeta(
  {
    samp: new VariableMeta("samp", VariableKind.Sampler, `sampler`),
    irradianceTex: new VariableMeta("irradianceTex", VariableKind.Texture, `texture_2d<f32>`),
    gAlbedo: new VariableMeta("gAlbedo", VariableKind.Texture, `texture_2d<f32>`),
    gEmission: new VariableMeta("gEmission", VariableKind.Texture, `texture_2d<f32>`),
    uComp: new VariableMeta("uComp", VariableKind.Uniform, `vec4<f32>`),
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

struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VOut {
  var o: VOut;
  o.pos = vec4f(POSITION[i], 0., 1.);
  o.uv = TEX[i];
  return o;
}

// uComp: .x = ambient floor
@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let a = textureSample(gAlbedo, samp, in.uv);
  if (a.a == 0.0) { discard; } // background -> leave swapchain clear
  let irr = textureSampleLevel(irradianceTex, samp, in.uv, 0.).rgb;
  let emis = textureSample(gEmission, samp, in.uv).rgb;
  let lit = emis + a.rgb * (uComp.x + irr);
  return vec4f(lit, 1.0);
}
  `,
);

// ============================================================================
// FACTORY
// ============================================================================
// Live-tunable RC params (the rest — CASCADE_COUNT, BASE_RAYS, probe spacing — are
// structural and baked). Defaults come from the constants above. Mutate `.params`
// on the returned function (e.g. from a lil-gui panel) and changes take effect next
// frame, since these all flow through per-frame uniform uploads.
export interface RCParams {
  marchSteps: number;
  zBias: number;
  ambient: number;
  ambientFill: number;
  baseIntervalPx: number;
  sky: [number, number, number];
  denoiseIterations: number; // à-trous passes; 0 = off
  denoiseWorldSigma: number;
  denoiseNormalPow: number;
}

export function defaultRCParams(): RCParams {
  return {
    marchSteps: MARCH_STEPS,
    zBias: ZBIAS_WORLD,
    ambient: AMBIENT,
    ambientFill: AMBIENT_FILL,
    baseIntervalPx: BASE_INTERVAL_PX,
    sky: [SKY[0], SKY[1], SKY[2]],
    denoiseIterations: DENOISE_ITERS,
    denoiseWorldSigma: DENOISE_WORLD_SIGMA,
    denoiseNormalPow: DENOISE_NORMAL_POW,
  };
}

export function createRadianceCascades(
  device: GPUDevice,
  rcTextures: { cascA: GPUTexture; cascB: GPUTexture },
  // Pass a stable params object to survive a factory rebuild (e.g. when the
  // cascade textures are resized live) — keeps GUI controllers pointed at it.
  externalParams?: RCParams,
) {
  const params: RCParams = externalParams ?? defaultRCParams();
  // Linear sampler for the cascade textures (bilerp merge + gather upsample).
  const linSamp = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });

  const { cascA, cascB } = rcTextures;

  // ---- cascade (raycast+merge) shader ----
  const cascShader = new GPUShader(cascadeMeta);
  cascadeMeta.uniforms.samp.setSampler(linSamp);
  // depth/gAlbedo/gEmission textures are STATIC across passes — set once.

  // ---- gather shader (cascade0 -> raw irradiance texture) ----
  const gatherShader = new GPUShader(gatherMeta);
  gatherMeta.uniforms.samp.setSampler(linSamp);

  // ---- à-trous denoise shader (irradiance ping-pong) ----
  const atrousShader = new GPUShader(atrousMeta);
  atrousMeta.uniforms.samp.setSampler(linSamp);

  // ---- composite shader (denoised irradiance + G-buffer -> lit) ----
  const compositeShader = new GPUShader(compositeMeta);
  compositeMeta.uniforms.samp.setSampler(linSamp);

  // CPU-side uniform scratch.
  const rcData = new Float32Array(20); // array<vec4f,5>
  const screenData = new Float32Array(4);
  const gatherData = new Float32Array(4);
  const atData = new Float32Array(4);
  const compData = new Float32Array(4);
  const compBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  // PER-ITERATION à-trous uniform buffers (same single-command-buffer aliasing rule
  // as rcBufs: each pass needs its own stride, written once).
  const atBufs: GPUBuffer[] = [];
  for (let i = 0; i < MAX_ATROUS; i++) {
    atBufs.push(
      device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    );
  }

  // Shared (per-frame) camera UBOs — same value across all cascade passes.
  const camBuf = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const vpBuf = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const camPosBuf = device.createBuffer({
    size: 16, // vec4 (xyz = camera world pos)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const camPosData = new Float32Array(4);
  const screenBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // PER-CASCADE uRC UBOs. CRITICAL: all CASCADE_COUNT cascade passes are encoded
  // into ONE command buffer and submitted together, so a single uRC buffer written
  // N times in the loop would alias (every pass reads the LAST write). One buffer
  // per cascade, each written once, sidesteps the aliasing entirely.
  const rcBufs: GPUBuffer[] = [];
  for (let n = 0; n < CASCADE_COUNT; n++) {
    rcBufs.push(
      device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    );
  }

  let cascPipeline: GPURenderPipeline | undefined;
  let gatherPipeline: GPURenderPipeline | undefined;
  let atrousPipeline: GPURenderPipeline | undefined;
  let compositePipeline: GPURenderPipeline | undefined;
  let bg0Layout: GPUBindGroupLayout | undefined;
  let gatherBindGroup: GPUBindGroup | undefined;
  // à-trous bind group per iteration: src alternates irA/irB, output is the other.
  // bind group i reads (i even? irA : irB) + atBufs[i]; we draw into the other.
  const atrousBindGroups: (GPUBindGroup | undefined)[] = new Array(MAX_ATROUS);
  // composite reads whichever irradiance buffer holds the final result — parity
  // depends on the live iteration count, so build one bind group per source.
  let compositeBGfromA: GPUBindGroup | undefined;
  let compositeBGfromB: GPUBindGroup | undefined;
  // Full-res HDR irradiance ping-pong: gather writes irA, à-trous bounces irA<->irB.
  let irA: GPUTexture | undefined;
  let irB: GPUTexture | undefined;
  // Per-cascade bind group (source texture + that cascade's own uRC). Built once.
  const cascBindGroups: (GPUBindGroup | undefined)[] = new Array(CASCADE_COUNT);
  // The texture cascade 0 writes into (parity is fixed at build time).
  let casc0Tex: GPUTexture = cascA;

  function init(
    gAlbedo: GPUTexture,
    gNormal: GPUTexture,
    gEmission: GPUTexture,
    depth: GPUTexture,
  ) {
    if (cascPipeline !== undefined) return;

    // cascade pass: rgba16float target, no blend.
    cascPipeline = cascShader.getRenderPipeline(device, "vs_main", "fs_main", {
      targets: [{ format: "rgba16float", blend: "none" }],
    });
    bg0Layout = cascShader.createBindGroupLayout(device, 0);

    // static texture bindings for the cascade pass (everything except cascadeIn).
    cascadeMeta.uniforms.depthTex.setTexture(depth);
    cascadeMeta.uniforms.gAlbedo.setTexture(gAlbedo);
    cascadeMeta.uniforms.gEmission.setTexture(gEmission);

    // Full-res HDR irradiance ping-pong buffers (gather writes irA; à-trous bounces).
    const mkIrr = () =>
      device.createTexture({
        size: [gAlbedo.width, gAlbedo.height, 1],
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
    irA = mkIrr();
    irB = mkIrr();

    // gather pass: rgba16float irradiance target (HDR, no albedo/emission combine).
    gatherPipeline = gatherShader.getRenderPipeline(device, "vs_main", "fs_main", {
      targetFormat: "rgba16float",
    });

    // à-trous pass: rgba16float irradiance target (ping-pong).
    atrousPipeline = atrousShader.getRenderPipeline(device, "vs_main", "fs_main", {
      targetFormat: "rgba16float",
    });
    atrousMeta.uniforms.depthTex.setTexture(depth);

    // composite pass: bgra8unorm full-size target.
    compositePipeline = compositeShader.getRenderPipeline(device, "vs_main", "fs_main", {
      targetFormat: "bgra8unorm",
    });

    // Precompute ping-pong parity + build a bind group per cascade. Walk top->down
    // (n = CASCADE_COUNT-1 .. 0): cascade n writes `dst` and reads `src` (= n+1's
    // result). The top cascade ignores cascadeIn (no merge), but still needs a valid
    // binding, so we bind the other texture harmlessly.
    let dst = cascA;
    let src = cascB;
    const depthView = depth.createView();
    const albedoView = gAlbedo.createView();
    const emissionView = gEmission.createView();
    for (let n = CASCADE_COUNT - 1; n >= 0; n--) {
      cascBindGroups[n] = device.createBindGroup({
        layout: bg0Layout,
        entries: [
          { binding: cascadeMeta.uniforms.samp.binding, resource: linSamp },
          { binding: cascadeMeta.uniforms.depthTex.binding, resource: depthView },
          { binding: cascadeMeta.uniforms.gAlbedo.binding, resource: albedoView },
          { binding: cascadeMeta.uniforms.gEmission.binding, resource: emissionView },
          { binding: cascadeMeta.uniforms.cascadeIn.binding, resource: src.createView() },
          { binding: cascadeMeta.uniforms.uInvViewProj.binding, resource: { buffer: camBuf } },
          { binding: cascadeMeta.uniforms.uViewProj.binding, resource: { buffer: vpBuf } },
          { binding: cascadeMeta.uniforms.uCamPos.binding, resource: { buffer: camPosBuf } },
          { binding: cascadeMeta.uniforms.uScreen.binding, resource: { buffer: screenBuf } },
          { binding: cascadeMeta.uniforms.uRC.binding, resource: { buffer: rcBufs[n] } },
        ],
      });
      if (n === 0) casc0Tex = dst;
      const t = dst;
      dst = src;
      src = t;
    }

    // gather bind group: casc0Tex parity is fixed, so build once.
    gatherMeta.uniforms.casc0.setTexture(casc0Tex);
    gatherBindGroup = device.createBindGroup({
      layout: gatherShader.createBindGroupLayout(device, 0),
      entries: [
        { binding: gatherMeta.uniforms.samp.binding, resource: linSamp },
        { binding: gatherMeta.uniforms.casc0.binding, resource: casc0Tex.createView() },
        {
          binding: gatherMeta.uniforms.uRCG.binding,
          resource: { buffer: gatherShader.uniforms.uRCG.getGPUBuffer(device) },
        },
      ],
    });

    // à-trous bind groups (reuse camBuf=invViewProj + screenBuf). Iteration i reads
    // (i even? irA : irB) and writes the other; each has its own stride uniform.
    const normalView = gNormal.createView();
    const atLayout = atrousShader.createBindGroupLayout(device, 0);
    const irAView = irA.createView();
    const irBView = irB.createView();
    for (let i = 0; i < MAX_ATROUS; i++) {
      atrousBindGroups[i] = device.createBindGroup({
        layout: atLayout,
        entries: [
          { binding: atrousMeta.uniforms.samp.binding, resource: linSamp },
          { binding: atrousMeta.uniforms.srcTex.binding, resource: i % 2 === 0 ? irAView : irBView },
          { binding: atrousMeta.uniforms.gNormal.binding, resource: normalView },
          { binding: atrousMeta.uniforms.depthTex.binding, resource: depthView },
          { binding: atrousMeta.uniforms.uInvViewProj.binding, resource: { buffer: camBuf } },
          { binding: atrousMeta.uniforms.uScreen.binding, resource: { buffer: screenBuf } },
          { binding: atrousMeta.uniforms.uAT.binding, resource: { buffer: atBufs[i] } },
        ],
      });
    }

    // composite bind groups — one per possible final-irradiance source (parity of
    // the live iteration count decides which at run time).
    const compLayout = compositeShader.createBindGroupLayout(device, 0);
    const mkComp = (irView: GPUTextureView) =>
      device.createBindGroup({
        layout: compLayout,
        entries: [
          { binding: compositeMeta.uniforms.samp.binding, resource: linSamp },
          { binding: compositeMeta.uniforms.irradianceTex.binding, resource: irView },
          { binding: compositeMeta.uniforms.gAlbedo.binding, resource: albedoView },
          { binding: compositeMeta.uniforms.gEmission.binding, resource: emissionView },
          { binding: compositeMeta.uniforms.uComp.binding, resource: { buffer: compBuf } },
        ],
      });
    compositeBGfromA = mkComp(irAView);
    compositeBGfromB = mkComp(irBView);
  }

  // Each cascade's output target follows the SAME parity as the bind-group build.
  function cascDst(n: number): GPUTexture {
    // dst starts at cascA for n=CASCADE_COUNT-1 and flips each step down.
    // flips so far = (CASCADE_COUNT-1 - n); even -> cascA, odd -> cascB.
    return (CASCADE_COUNT - 1 - n) % 2 === 0 ? cascA : cascB;
  }

  function uploadCamera(fullW: number, fullH: number) {
    device.queue.writeBuffer(camBuf, 0, invViewProjection as Float32Array);
    device.queue.writeBuffer(vpBuf, 0, projectionMatrix as Float32Array);
    camPosData[0] = cameraPosition[0];
    camPosData[1] = cameraPosition[1];
    camPosData[2] = cameraPosition[2];
    device.queue.writeBuffer(camPosBuf, 0, camPosData);
    // uScreen: depth is sampled at FULL G-buffer res (crisp occlusion).
    screenData[0] = fullW;
    screenData[1] = fullH;
    screenData[2] = 1 / fullW;
    screenData[3] = 1 / fullH;
    device.queue.writeBuffer(screenBuf, 0, screenData);
  }

  function uploadCascade(n: number, rcW: number, rcH: number) {
    rcData.fill(0);
    rcData[0] = rcW;
    rcData[1] = rcH;
    rcData[2] = 1 / rcW;
    rcData[3] = 1 / rcH;
    rcData[4] = n;
    rcData[5] = rayCount(n);
    rcData[6] = raysPerSide(n);
    rcData[7] = probeSpacing(n);
    rcData[8] = n + 1 < CASCADE_COUNT ? probeSpacing(n + 1) : probeSpacing(n) * 2;
    // tNear/tFar derive from the live base interval (doubling shell per cascade).
    const base = params.baseIntervalPx;
    rcData[9] = n === 0 ? 0 : base * (2 ** n - 1);
    rcData[10] = base * (2 ** (n + 1) - 1);
    rcData[11] = n === CASCADE_COUNT - 1 ? 1 : 0; // isTop
    rcData[12] = 0.1; // near (unused in screen-space test, kept for parity)
    rcData[13] = 20000; // far
    rcData[14] = params.zBias;
    rcData[15] = params.ambientFill;
    rcData[16] = params.marchSteps;
    rcData[17] = params.sky[0];
    rcData[18] = params.sky[1];
    rcData[19] = params.sky[2];
    device.queue.writeBuffer(rcBufs[n], 0, rcData);
  }

  const run = function radianceCascades(
    encoder: GPUCommandEncoder,
    gAlbedo: GPUTexture,
    gNormal: GPUTexture,
    gEmission: GPUTexture,
    depth: GPUTexture,
    out: GPUTexture,
  ) {
    init(gAlbedo, gNormal, gEmission, depth);

    const rcW = cascA.width;
    const rcH = cascA.height;
    const fullW = gAlbedo.width;
    const fullH = gAlbedo.height;

    uploadCamera(fullW, fullH);

    // March top->down, ping-pong. cascadeIn for cascade n reads cascade n+1's result;
    // each cascade has its OWN uRC buffer + bind group (built in init) to avoid
    // per-pass uniform aliasing within the single submitted command buffer.
    for (let n = CASCADE_COUNT - 1; n >= 0; n--) {
      uploadCascade(n, rcW, rcH);

      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: cascDst(n).createView(),
            clearValue: [0, 0, 0, 0],
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      pass.setPipeline(cascPipeline!);
      pass.setBindGroup(0, cascBindGroups[n]!);
      pass.draw(6, 1, 0, 0);
      pass.end();
    }

    // ---- gather: cascade0 -> raw irradiance (rgba16float) ----
    gatherData[0] = rcW;
    gatherData[1] = rcH;
    gatherData[2] = raysPerSide(0);
    gatherData[3] = 0;
    device.queue.writeBuffer(gatherShader.uniforms.uRCG.getGPUBuffer(device), 0, gatherData);

    const gpass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: irA!.createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    gpass.setPipeline(gatherPipeline!);
    gpass.setBindGroup(0, gatherBindGroup!);
    gpass.draw(6, 1, 0, 0);
    gpass.end();

    // ---- à-trous denoise: N edge-aware passes, stride doubling, ping-pong irA<->irB.
    // Iteration i reads (i even? irA : irB), writes the other. worldSigma is scaled by
    // stride so the world-space reach stays consistent as the kernel widens.
    const iters = Math.min(MAX_ATROUS, Math.max(0, Math.floor(params.denoiseIterations)));
    for (let i = 0; i < iters; i++) {
      const stride = 1 << i; // 1,2,4,8,...
      atData[0] = stride;
      atData[1] = params.denoiseWorldSigma * stride;
      atData[2] = params.denoiseNormalPow;
      atData[3] = 0;
      device.queue.writeBuffer(atBufs[i], 0, atData);

      const apass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: (i % 2 === 0 ? irB! : irA!).createView(),
            clearValue: [0, 0, 0, 0],
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      apass.setPipeline(atrousPipeline!);
      apass.setBindGroup(0, atrousBindGroups[i]!);
      apass.draw(6, 1, 0, 0);
      apass.end();
    }
    // Final irradiance lives in irA when iters is even (incl. 0), irB when odd.
    const finalFromA = iters % 2 === 0;

    // ---- composite: lit = emission + albedo*(ambient + irradiance) (bgra8unorm) ----
    compData[0] = params.ambient;
    device.queue.writeBuffer(compBuf, 0, compData);

    const cpass = encoder.beginRenderPass({
      colorAttachments: [
        { view: out.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" },
      ],
    });
    cpass.setPipeline(compositePipeline!);
    cpass.setBindGroup(0, finalFromA ? compositeBGfromA! : compositeBGfromB!);
    cpass.draw(6, 1, 0, 0);
    cpass.end();
  };

  // Expose the live params so a debug GUI can mutate them (applied next frame).
  return Object.assign(run, { params });
}
