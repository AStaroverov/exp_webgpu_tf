import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

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
// PERF: this pass runs at HALF resolution (¼ the pixels → ~4× less cone work). The full-res
// G-buffer is still sampled (at the half-res pixel ·2, clamped); the composite bilinear-
// upsamples this output back to full res — indirect light is low-frequency, so that is fine.
//
// IMPORTANCE-SAMPLED HEMISPHERE GATHER. Rather than brute-forcing many random hemisphere
// cones to find the lights, we AIM one narrow cone straight at each emitter (where the
// radiance is concentrated) for a sharp, far-reaching soft shadow, and keep a few WIDE
// Fibonacci fill cones for the diffuse bounce / ambient term. Both sets march the SAME
// trace_cone through the voxelRadiance pyramid with front-to-back occlusion, and accumulate
// into ONE cosine-weighted hemisphere integral (acc / occAcc / wsum) — it is NOT a separate
// analytic light, just a smarter sampling of the existing cone gather. The aimed directions
// are deterministic, so they carry no jitter banding ("denim"); a handful of cones replaces
// the old 48, killing both the banding and the lag.
//
// This is what produces real color bleeding. The earlier single-cone-along-the-normal form
// was the Layer-2 INTERMEDIATE (a bent-normal / AO preview); the hemisphere gather here is
// directly comparable to the brute-force gi reference, which is also a cosine-weighted
// hemisphere average.
//
// Mirrors voxelDebug.shader.ts for the fullscreen setup + `unproject` + reverse-Z NDC
// reconstruction. textureSampleLevel (explicit LOD) is used in the trace loop — legal in
// non-uniform control flow (unlike textureSample).

export const shaderMeta = new ShaderMeta(
  {
    // .x = normalBias, .y = maxDist (world), .z = aperture = tan(halfAngle), .w = giStrength.
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<f32>`),
    // .x = screen width (px), .y = screen height (px), .z = fill-cone count (angular
    // resolution of the ambient/bounce term), .w = active light count (0..8).
    params2: new VariableMeta("uParams2", VariableKind.Uniform, `vec4<f32>`),
    // Emitters to importance-sample: .xyz = world CENTER, .w = radius (penumbra source).
    // Only the first i32(uParams2.w) entries are live.
    lights: new VariableMeta("lights", VariableKind.Uniform, `array<vec4<f32>, 8>`),
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
    // .x = uAniso (1 = anisotropic VCT path, 0 = isotropic) — A/B anti-leak toggle. yzw spare.
    aniso: new VariableMeta("uAniso", VariableKind.Uniform, `vec4<f32>`),
    // The voxelRadiance mip pyramid (ALL mips) — the ISOTROPIC path + the lod<1 lerp read it.
    voxelRadiance: new VariableMeta("voxelRadiance", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    // The 6 ANISOTROPIC directional radiance volumes (±X,±Y,±Z), ALL-mips views. The cone picks
    // the 3 facing back toward its direction and blends them (Crassin). Start at iso mip1 res.
    voxelDirNegX: new VariableMeta("voxelNegX", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    voxelDirPosX: new VariableMeta("voxelPosX", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    voxelDirNegY: new VariableMeta("voxelNegY", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    voxelDirPosY: new VariableMeta("voxelPosY", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    voxelDirNegZ: new VariableMeta("voxelNegZ", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    voxelDirPosZ: new VariableMeta("voxelPosZ", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    // Filtering sampler for textureSampleLevel over the pyramids (shared by iso + the 6 dir vols).
    voxelSampler: new VariableMeta("voxelSampler", VariableKind.Sampler, `sampler`),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
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

// Clamp the trace LOD to dodge high-mip glitch (the iso pyramid has 8 levels; the directional
// pyramids only ~7 at base/2, so this is also a bounds guard for the aniso path).
const MAX_LOD = 6.0;

// Sample the radiance volume(s) in cone direction 'dir' at filtered LOD 'lod' (iso-pyramid LOD).
// Returns PREMULTIPLIED (rgb=radiance·coverage, a=coverage), the same convention as the iso path.
//   uAniso < 0.5  → isotropic path: a single textureSampleLevel(voxelRadiance,...) — byte-
//                   identical to the original cone (the A side of the A/B anti-leak test).
//   else          → anisotropic path: pick the 3 sign-selected directional volumes (the faces
//                   looking back toward the cone), weight by dir·dir (sums to 1 for a normalized
//                   dir → a proper convex blend, Crassin canonical), and sample each. The
//                   directional volumes start at base/2 = iso mip1, so their LOD axis is shifted
//                   by 1 (aLod = lod-1). For lod<1 the directional level-0 does not exist yet, so
//                   sample the ISO base and lerp into the aniso result across [0,1].
// Sign/face convention (rdinse): dir.x<0 → the -X volume, else +X (the volume whose front-to-back
// integration sees the cone-facing voxels as "near"); same for Y/Z. Must match the downsample.
// textureSampleLevel (explicit f32 LOD) is legal in the non-uniform sign-selected branches
// (unlike textureSample).
fn sampleRadiance(uvw: vec3<f32>, lod: f32, dir: vec3<f32>) -> vec4<f32> {
  if (uAniso.x < 0.5) {
    return textureSampleLevel(voxelRadiance, voxelSampler, uvw, lod);
  }

  let aLod = max(lod - 1.0, 0.0);
  let w = dir * dir;
  var aniso = vec4<f32>(0.0);
  if (dir.x < 0.0) { aniso = aniso + w.x * textureSampleLevel(voxelNegX, voxelSampler, uvw, aLod); }
  else            { aniso = aniso + w.x * textureSampleLevel(voxelPosX, voxelSampler, uvw, aLod); }
  if (dir.y < 0.0) { aniso = aniso + w.y * textureSampleLevel(voxelNegY, voxelSampler, uvw, aLod); }
  else            { aniso = aniso + w.y * textureSampleLevel(voxelPosY, voxelSampler, uvw, aLod); }
  if (dir.z < 0.0) { aniso = aniso + w.z * textureSampleLevel(voxelNegZ, voxelSampler, uvw, aLod); }
  else            { aniso = aniso + w.z * textureSampleLevel(voxelPosZ, voxelSampler, uvw, aLod); }

  if (lod < 1.0) {
    let iso = textureSampleLevel(voxelRadiance, voxelSampler, uvw, lod);
    return mix(iso, aniso, clamp(lod, 0.0, 1.0));
  }
  return aniso;
}

// One cone marched along dir from origin: diameter grows with distance, each step samples
// voxelRadiance at LOD=log2(diameter/voxelSize) and composites front-to-back ("over").
// reach = how far this cone marches (world units). The step is floored at reach/64 so the
// cone ALWAYS spans its full reach within the 64-step budget — without this, a narrow cone's
// tiny near-field steps burn the budget and it dies ~16 units out (the "light stops at cone
// reach" problem). fadeFrac>0 tapers the gathered radiance over the last fadeFrac of reach
// (hides the artificial cutoff of the fill cones); pass 0 for aimed cones, which end at the
// real light, so they must NOT be tapered. Occlusion (alpha) is never tapered → shadows stay.
fn trace_cone(origin: vec3<f32>, dir: vec3<f32>, aperture: f32, reach: f32, fadeFrac: f32, startJ: f32) -> vec4<f32> {
  var col = vec3<f32>(0.0);
  var alpha = 0.0;
  let voxelSize = uGridOrigin.w;
  let gridMin = uGridOrigin.xyz;
  let extent = vec3<f32>(uGridDims.xyz) * voxelSize;
  let stepFloor = reach / 64.0;
  // Per-pixel SCALE of the start distance (startJ in [0,1)). Because the march is ~geometric
  // (step grows with distance), scaling dist0 shifts the sampling "shells" by a per-pixel
  // factor at EVERY distance → the concentric "tree-ring" banding around bright sources
  // dithers across pixels and is blurred away by the half-res upsample. Lets few cones look
  // smooth without going to 48.
  var dist = voxelSize * (1.0 + startJ);
  for (var i = 0; i < 64; i = i + 1) {
    if (alpha >= 1.0 || dist > reach) { break; }
    let diameter = max(voxelSize, 2.0 * aperture * dist);
    let lod = min(log2(diameter / voxelSize), MAX_LOD);
    let wp = origin + dir * dist;
    let uvw = (wp - gridMin) / extent;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) { break; }
    let s = sampleRadiance(uvw, lod, dir);
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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  // This pass renders at HALF res; map the half-res framebuffer coord to a representative
  // full-res G-buffer texel (clamped). uParams2.xy carries the FULL canvas dims.
  let half = vec2<i32>(floor(input.position.xy));
  let full = min(half * 2, vec2<i32>(uParams2.xy) - vec2<i32>(1));

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
  let origin = P + N * (cellSize * 1.5 + uParams.x);

  // N-cone cosine-weighted diffuse hemisphere gather around N, on a procedural Fibonacci
  // (golden-angle) hemisphere. The cone count (uParams2.z) is the angular resolution: per-
  // direction occupancy (cone.a) IS the shadow term, so more cones = sharper shadows.
  let basis = build_basis(N);
  let aperture = uParams.z;
  let count = max(1, i32(uParams2.z));
  // Per-pixel azimuth rotation of the whole cone set (interleaved-gradient-noise hash) so the
  // gaps between narrow cones fall in DIFFERENT directions each pixel → fixed "ray" banding
  // becomes fine dither, which the half-res + bilinear upsample then blurs away. Lets a narrow
  // aperture use FEWER cones without visible streaks.
  let jitter = fract(52.9829189 * fract(dot(vec2<f32>(half), vec2<f32>(0.06711056, 0.00583715)))) * 6.2831853;
  // Decorrelated [0,1) per-pixel value for radial (start-distance) dither — kills the
  // concentric "tree-ring" banding around bright sources at low cone counts.
  let jrad = fract(jitter * 1.61803399);

  var acc = vec3<f32>(0.0);
  // occAcc accumulates the SAME cosine-weighted average over each cone's opacity (.a) — the
  // hemisphere "how blocked am I" measure that becomes ambient occlusion / soft shadows.
  var occAcc = 0.0;
  var wsum = 0.0;

  // (a) AIMED cones — one narrow cone per emitter, pointed straight at the light center for a
  // sharp, far-reaching direct + soft-shadow term. Its aperture = the light's angular size
  // (radius / distance) sets the penumbra width, and the cosine term (ndl) is its weight, so
  // these fold into the SAME hemisphere integral as the fill cones below.
  let lc = min(8, max(0, i32(uParams2.w)));
  for (var j = 0; j < lc; j = j + 1) {
    let toL = lights[j].xyz - origin;
    let d = length(toL);
    if (d < 1e-3) { continue; }
    let dir = toL / d;
    let ndl = max(dot(N, dir), 0.0);
    if (ndl <= 0.0) { continue; }
    let ap = clamp(lights[j].w / d, 0.02, 0.5);   // angular size of the light = penumbra width
    // reach = distance to the light so the cone always arrives at the emitter (light reaches
    // as far as the source is, within the grid); no fade (it ends at the real light).
    let r = trace_cone(origin, dir, ap, d, 0.0, jrad);
    acc = acc + ndl * r.rgb;
    occAcc = occAcc + ndl * r.a;
    wsum = wsum + ndl;
  }

  // (b) FILL cones — the wide Fibonacci hemisphere set for soft ambient / bounce, accumulated
  // into the SAME acc/occAcc/wsum with cosine weights.
  for (var i = 0; i < count; i = i + 1) {
    let k = (f32(i) + 0.5) / f32(count);
    let cosT = 1.0 - k;                       // hemisphere: cosTheta in (0,1]
    let sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    let phi = f32(i) * 2.39996323 + jitter;   // golden angle + per-pixel rotation
    let local = vec3<f32>(sinT * cos(phi), sinT * sin(phi), cosT);
    let dir = normalize(basis * local);
    let w = cosT;                             // cosine weight
    // Fill/bounce cones reach the global maxDist (uParams.y) and softly fade over the last
    // quarter so their artificial cutoff doesn't show as a circle.
    let r = trace_cone(origin, dir, aperture, uParams.y, 0.25, jrad);
    acc = acc + w * r.rgb;
    occAcc = occAcc + w * r.a;
    wsum = wsum + w;
  }

  // Cosine-weighted average radiance + visibility (= 1 - occlusion, the AO / soft-shadow term).
  let irradiance = acc / max(wsum, 1e-4);
  let visibility = clamp(1.0 - occAcc / max(wsum, 1e-4), 0.0, 1.0);
  // rgb = indirect radiance·giStrength (the "cone" present mode shows this unchanged);
  // a = hemisphere visibility, read as AO by the composite.
  return vec4f(irradiance * uParams.w, visibility);
}
`,
);
