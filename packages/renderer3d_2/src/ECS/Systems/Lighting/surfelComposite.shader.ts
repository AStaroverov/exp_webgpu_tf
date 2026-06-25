import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { SURFEL_CAP, SURFEL_DIR0_W, SURFEL_DIR_COUNT } from "./surfelResources.ts";

// Surfel Radiance Cascades — Stage C COMPOSITE pass (doc §7).
//
// Fullscreen draw over the canvas-sized surfelLitTexture. SINGLE cascade, no merge.
// For each screen pixel we:
//   1. Reconstruct world position from the reverse-Z depth buffer + the inverse
//      view-projection (matching ResizeSystem.viewProjMatrix exactly — VERBATIM from
//      worldComposite). Background (normal.a < 0.5) -> scene * ambient passthrough.
//   2. Hash world.xyz into the SAME spatial-hash bucket Stage B insert/spawn use
//      (hash3 + cellSize), read the bucket's surfel ids (atomicLoad over surfel_grid).
//   3. For each nearby LIVE surfel within surfelSearchRadius, integrate its 16-dir
//      octahedral radiance tile against the receiver normal N (cosine-weighted, full-
//      sphere solid-angle normalization 4*PI / DIR_COUNT — identical to worldComposite),
//      then proximity-weighted blend (w = 1/(d²+eps)) across the bucket's surfels.
//   4. lit = albedo * (ambient + radiance).
//
// COORDINATES: world is Z-up, footprints in XY. The hash3 + cellSize MUST match
// surfelInsert.shader.ts / surfelSpawn.shader.ts so composite reads the same buckets
// the insert pass wrote.
//
// surfel_posr / surfel_norw / surfel_rad / surfel_grid are STANDALONE GPUBuffers
// (see surfelResources.ts); the VariableMetas below exist ONLY for WGSL emission +
// the (kind-based, size-agnostic) bind-group layout. surfel_grid is atomic-typed, so
// it MUST be read_write (StorageWrite) and read via atomicLoad — that forces it to
// group 2, making groups non-contiguous {0,1,2}. The consuming system therefore
// builds this RENDER pipeline with autoLayout + manual bind groups (NOT an explicit
// pipeline layout, which mis-packs non-contiguous groups). Bind the storage buffers
// manually against pipeline.getBindGroupLayout(g).

// Octahedral tile side — MUST match worldGather / surfelGather and the surfel_rad cache.
const DIR0_W = SURFEL_DIR0_W;

// FRAGMENT-only group-0 uniform (the fullscreen vertex stage uses NO uniforms, so
// keeping them FRAGMENT-visible stays well under the 12-uniform-per-stage limit).
const uF = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, {
    visibility: GPUShaderStage.FRAGMENT,
  });

export const shaderMeta = new ShaderMeta(
  {
    // ---- group 0 : textures + sampler (FRAGMENT) ----
    inputSampler: new VariableMeta("textureSampler", VariableKind.Sampler, `sampler`, {
      visibility: GPUShaderStage.FRAGMENT,
    }),
    // Albedo / main-pass color the light multiplies over.
    sceneTexture: new VariableMeta("sceneTexture", VariableKind.Texture, `texture_2d<f32>`, {
      visibility: GPUShaderStage.FRAGMENT,
    }),
    // G-buffer world normals (packed *0.5+0.5; a = surface mask) from the main pass.
    normalTexture: new VariableMeta("normalTexture", VariableKind.Texture, `texture_2d<f32>`, {
      visibility: GPUShaderStage.FRAGMENT,
    }),
    // Reverse-Z depth (depth32float) sampled as a depth texture (textureLoad only).
    // sampleType MUST be "depth" — depth32float is not a filterable float, so the
    // bind-group layout entry has to advertise Depth (else CreateBindGroup rejects it).
    depthTexture: new VariableMeta("depthTexture", VariableKind.Texture, `texture_depth_2d`, {
      textureSampleType: "depth",
      visibility: GPUShaderStage.FRAGMENT,
    }),

    // ---- group 0 : uniforms (FRAGMENT-only) ----
    // Spatial-hash cell size in world units — MUST equal the insert/spawn cellSize
    // (Stage B derives it from 2*radius). Hashing world.xyz with the SAME cellSize
    // reads the SAME buckets the insert pass filled.
    cellSize: uF("uCellSize", `f32`),
    // Octahedral tile side as u32 (== SURFEL_DIR0_W), for the cosine-integral loops.
    dir0W: uF("uDir0W", `u32`),
    // Stock omni light floor (matches worldComposite/overlay uAmbient).
    ambient: uF("uAmbient", `f32`),
    // World-space search radius for accepting a bucket surfel (~cellSize). Surfels
    // farther than this from the receiver are skipped (the bucket can hold surfels
    // from a neighboring world cell once the disc overlaps a cell boundary).
    surfelSearchRadius: uF("uSurfelSearchRadius", `f32`),
    // Inverse view-projection (gl-matrix mat4, column-major) for world-pos
    // reconstruction. MUST equal inverse(ResizeSystem.viewProjMatrix) (reverse-Z,
    // Z-up, NEAR->1 FAR->0). Demo supplies the inverse each frame.
    invViewProj: uF("uInvViewProj", `mat4x4<f32>`),

    // ---- group 1 : surfel storage reads (StorageRead => @group(1)) ----
    // Declaration order = binding order: surfel_posr (0), surfel_norw (1), surfel_rad (2).
    // surfel_posr: xyz = world position, w = radius² (w == 0 => DEAD slot).
    surfelPosr: new VariableMeta(
      "surfel_posr",
      VariableKind.StorageRead,
      `array<vec4<f32>, ${SURFEL_CAP}>`,
      { visibility: GPUShaderStage.FRAGMENT },
    ),
    // surfel_norw: xyz = surface normal, w = recycle marker (unused here).
    surfelNorw: new VariableMeta(
      "surfel_norw",
      VariableKind.StorageRead,
      `array<vec4<f32>, ${SURFEL_CAP}>`,
      { visibility: GPUShaderStage.FRAGMENT },
    ),
    // surfel_rad: vec4<f32>[CAP * DIR_COUNT] — the gather pass's radiance cache.
    // rgb = direction radiance, a = visibility. Index id*DIR_COUNT + (v*DIR0_W + u).
    surfelRad: new VariableMeta(
      "surfel_rad",
      VariableKind.StorageRead,
      `array<vec4<f32>, ${SURFEL_CAP * SURFEL_DIR_COUNT}>`,
      { visibility: GPUShaderStage.FRAGMENT },
    ),

    // ---- group 2 : spatial hash grid (StorageWrite => @group(2)) ----
    // atomic<u32> array — MUST stay read_write (no write-only mode) and is read via
    // atomicLoad here. Per bucket: [count, id0..id(CELL_K-1)], stride 1 + CELL_K.
    // Length GRID_CAP*(1+CELL_K) = 65536*17 = 1114112 (see surfelResources.ts).
    surfelGrid: new VariableMeta(
      "surfel_grid",
      VariableKind.StorageWrite,
      `array<atomic<u32>, 1114112>`,
      { visibility: GPUShaderStage.FRAGMENT },
    ),
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

// V grows DOWNWARD (top row = 0) — same convention as worldComposite/overlay.
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

const PI: f32 = 3.14159265;
const DIR0_W: u32 = ${DIR0_W}u;
const DIR_COUNT: u32 = ${SURFEL_DIR_COUNT}u;

// Grid layout constants — MUST match surfelResources.ts / surfelInsert.shader.ts.
const GRID_CAP: u32 = 65536u;
const CELL_K: u32 = 16u;
const GRID_STRIDE: u32 = 1u + CELL_K; // [count, id0..id(CELL_K-1)]

// ===== Octahedral decode (doc §3.4, verbatim with gather/worldComposite) =====
fn oct_decode(e_in: vec2<f32>) -> vec3<f32> {
  let e = e_in;
  var v = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
  if (v.z < 0.0) {
    v = vec3<f32>((1.0 - abs(v.yx)) * sign(v.xy), v.z);
  }
  return normalize(v);
}

// Solid integer spatial hash (Teschner et al. primes) — VERBATIM from
// surfelInsert.shader.ts. bitcast i32->u32 so negative cell coords hash correctly.
fn hash3(c: vec3<i32>) -> u32 {
  return (bitcast<u32>(c.x) * 73856093u)
       ^ (bitcast<u32>(c.y) * 19349663u)
       ^ (bitcast<u32>(c.z) * 83492791u);
}

// Integrate one surfel's octahedral radiance tile against the receiver normal N:
// sum the tile's radiance weighted by Lambert max(0, dot(N, dir)), normalized by the
// per-cell solid angle (4*PI / DIR_COUNT). Identical math to worldComposite's
// integrate_probe, but the radiance comes from the surfel_rad BUFFER (index
// sid*DIR_COUNT + v*DIR0_W + u) instead of a probe texture.
fn integrate_surfel(sid: u32, N: vec3<f32>) -> vec3<f32> {
  let base = sid * DIR_COUNT;
  var acc = vec3<f32>(0.0);
  for (var v: u32 = 0u; v < DIR0_W; v = v + 1u) {
    for (var u: u32 = 0u; u < DIR0_W; u = u + 1u) {
      let oct = ((vec2<f32>(f32(u), f32(v)) + 0.5) / f32(DIR0_W)) * 2.0 - 1.0;
      let dir = oct_decode(oct);
      let cosw = max(0.0, dot(N, dir));
      acc = acc + surfel_rad[base + v * DIR0_W + u].rgb * cosw;
    }
  }
  // Full-sphere solid-angle normalization (DIR_COUNT == DIR0_W*DIR0_W).
  return acc * (4.0 * PI / f32(DIR_COUNT));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let scene = textureSample(sceneTexture, textureSampler, input.texCoord);

  let px = vec2<i32>(floor(input.position.xy));
  let n = textureLoad(normalTexture, px, 0);
  if (n.a < 0.5) {
    // No surface (background): stock omni floor, no surfel light.
    return vec4f(scene.rgb * uAmbient, scene.a);
  }

  // --- Reconstruct world position from reverse-Z depth (VERBATIM worldComposite) ---
  // WebGPU clip Y is UP; texCoord V grows DOWN, so NDC.y = (1 - v)*2 - 1.
  // depth is the stored reverse-Z value (NEAR->1, FAR->0); feed it straight as NDC z.
  let depth = textureLoad(depthTexture, px, 0);
  let ndc = vec4<f32>(input.texCoord.x * 2.0 - 1.0, (1.0 - input.texCoord.y) * 2.0 - 1.0, depth, 1.0);
  let wp4 = uInvViewProj * ndc;
  let world = wp4.xyz / wp4.w;

  let N = normalize(n.rgb * 2.0 - 1.0);

  // --- Gather the 4 NEAREST surfels over a 3x3x3 cell neighborhood ---
  // Single-bucket + 1/d² weighting was grainy: few surfels per cell, the weight
  // spikes on the nearest (≈ pick-nearest, hard), and the set jumps at cell borders.
  // Instead scan the neighborhood of cells covering [world ± R], keep the 4 nearest
  // surfels (deduped — a surfel sits in several cells), and blend them with a SMOOTH
  // falloff (0 at R). This smooths both the spatial discontinuities and the weights.
  let R = uSurfelSearchRadius;
  // Scan a FIXED 3³ neighbourhood (±1 cell) around the point's cell — cheap, bounded
  // cost (27 buckets · CELL_K). NOTE: the scan range and the falloff radius R are
  // DECOUPLED. The ±1 cells already hold several surfels (spacing ≈ cellSize); R only
  // controls the blend FALLOFF, and must exceed the spacing (default 2·cellSize) so
  // those neighbours get nonzero weight — otherwise the nearest-4 blend collapses to
  // nearest-1. (Scanning ±2 to "reach further" is what made the composite crawl.)
  let cc = vec3<i32>(floor(world / uCellSize));
  let span = 1;

  var bestSid = array<u32, 4>(0u, 0u, 0u, 0u);
  var bestD = array<f32, 4>(1e30, 1e30, 1e30, 1e30);

  for (var cz = cc.z - span; cz <= cc.z + span; cz = cz + 1) {
    for (var cy = cc.y - span; cy <= cc.y + span; cy = cy + 1) {
      for (var cx = cc.x - span; cx <= cc.x + span; cx = cx + 1) {
        let gbase = (hash3(vec3<i32>(cx, cy, cz)) % GRID_CAP) * GRID_STRIDE;
        let cnt = min(atomicLoad(&surfel_grid[gbase]), CELL_K);
        for (var k: u32 = 0u; k < cnt; k = k + 1u) {
          let sid = atomicLoad(&surfel_grid[gbase + 1u + k]);
          let sp = surfel_posr[sid];
          if (sp.w <= 0.0) { continue; } // dead slot (recycled since insert)
          var d = distance(sp.xyz, world);
          if (d >= R || d >= bestD[3]) { continue; } // too far / not in the top 4
          // Skip if this surfel is already one of the kept nearest (it's in many cells).
          if (sid == bestSid[0] || sid == bestSid[1] || sid == bestSid[2] || sid == bestSid[3]) {
            continue;
          }
          // Insertion-sort into the 4 nearest (ascending distance).
          var cand = sid;
          for (var i = 0; i < 4; i = i + 1) {
            if (d < bestD[i]) {
              let td = bestD[i]; bestD[i] = d; d = td;
              let ts = bestSid[i]; bestSid[i] = cand; cand = ts;
            }
          }
        }
      }
    }
  }

  // Blend the kept nearest surfels with a smooth falloff weight (0 at R).
  var radiance = vec3<f32>(0.0);
  var wsum: f32 = 0.0;
  for (var i = 0; i < 4; i = i + 1) {
    if (bestD[i] >= R) { continue; }
    let t = 1.0 - bestD[i] / R;
    let w = t * t; // smooth, → 0 at the search radius
    radiance = radiance + integrate_surfel(bestSid[i], N) * w;
    wsum = wsum + w;
  }
  if (wsum > 0.0) {
    radiance = radiance / wsum;
  } else {
    radiance = vec3<f32>(0.0);
  }

  // Diffuse GI: albedo * (ambient floor + cosine-integrated incoming radiance).
  let lit = scene.rgb * (uAmbient + radiance);
  return vec4f(lit, scene.a);
}
    `,
);
