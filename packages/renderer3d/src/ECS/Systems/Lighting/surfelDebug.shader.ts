import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { SURFEL_CAP } from "./surfelResources.ts";

// Surfel Radiance Cascades — Stage A DEBUG DRAW pass (doc, direct_draw idea).
//
// An INSTANCED draw of SURFEL_CAP instances, 6 verts each — a camera-facing
// screen-space billboard (a small round dot of constant pixel size, regardless
// of camera zoom). Lets us SEE the surfels sitting on visible surfaces.
//
// Per instance we read surfel_posr[instance] (xyz = world pos, w = radius²; w==0
// ⇒ DEAD slot) and surfel_norw[instance] (xyz = world normal). For a dead slot we
// emit an off-screen / clipped vertex so the instance draws nothing. For a live
// slot we project the surfel CENTER through uViewProj, then offset the 6 quad
// corners in CLIP space by quadPx / resolution (a constant screen-size dot — no
// world-space scaling, so zoom doesn't change the dot size). The corner uv is
// passed through for a round mask; the color visualizes the surfel orientation
// (normal * 0.5 + 0.5).
//
// BINDINGS (kind-derived by setupVariable; groups are contiguous {0,1} so the
// consuming system can use an EXPLICIT pipeline layout):
//   group 0 (uniforms, VERTEX|FRAGMENT):
//     binding 0  uViewProj : mat4x4<f32>  — ResizeSystem.viewProjMatrix (column-major)
//     binding 1  uParams   : vec4<f32>    — (resW, resH, quadPx, _)
//   group 1 (StorageRead, VERTEX):
//     binding 0  surfel_posr : array<vec4<f32>, CAP>  — xyz pos, w = radius² (0 ⇒ dead)
//     binding 1  surfel_norw : array<vec4<f32>, CAP>  — xyz normal, w = recycle marker
//
// Color target: bgra8unorm (matches both worldLitTexture and frame.renderTexture).
// No depth attachment — surfels draw on top of the lit scene (debug only).

export const shaderMeta = new ShaderMeta(
  {
    // ---- group 0 : per-frame uniforms ----
    // Forward view-projection (gl-matrix mat4, column-major) — same matrix
    // ResizeSystem maintains and the composite inverts.
    viewProj: new VariableMeta("uViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    // (resW, resH, quadPx, _) packed into one vec4 to stay frugal on uniforms.
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<f32>`),

    // ---- group 1 : surfel storage (StorageRead) ----
    // Bound to the STANDALONE surfel buffers (createSurfelResources). The buffer
    // byte size is fixed in JS; this declaration is only for WGSL emission + the
    // (size-agnostic) bind-group layout entry.
    posr: new VariableMeta(
      "surfel_posr",
      VariableKind.StorageRead,
      `array<vec4<f32>, ${SURFEL_CAP}>`,
      { visibility: GPUShaderStage.VERTEX },
    ),
    norw: new VariableMeta(
      "surfel_norw",
      VariableKind.StorageRead,
      `array<vec4<f32>, ${SURFEL_CAP}>`,
      { visibility: GPUShaderStage.VERTEX },
    ),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
// Unit-quad corners in [-1,1], two triangles (6 verts) — the per-corner offset
// is applied in CLIP space scaled by quadPx/resolution.
const CORNER = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f( 1.0,  1.0),
    vec2f(-1.0, -1.0),
    vec2f( 1.0,  1.0),
    vec2f(-1.0,  1.0)
  );

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,       // corner offset in [-1,1] for the round mask
  @location(1) color: vec3f,    // normal * 0.5 + 0.5
};

@vertex
fn vs_main(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
  var output: VertexOutput;

  let pr = surfel_posr[instanceIndex];

  // Dead slot (w == radius² == 0): emit a clipped vertex so it draws nothing.
  if (pr.w == 0.0) {
    output.position = vec4f(2.0, 2.0, 2.0, 1.0);
    output.uv = vec2f(0.0, 0.0);
    output.color = vec3f(0.0, 0.0, 0.0);
    return output;
  }

  // Project the surfel CENTER through the forward view-projection.
  let clip = uViewProj * vec4f(pr.xyz, 1.0);

  // Constant screen-size dot: offset the corner in CLIP space by quadPx/res.
  // Multiplying by clip.w keeps the post-perspective-divide offset = quadPx px
  // (here ortho => w==1, but this is robust either way).
  let corner = CORNER[vertexIndex];
  let res = uParams.xy;
  let quadPx = uParams.z;
  let offset = (corner * quadPx / res) * clip.w;

  output.position = vec4f(clip.xy + offset, clip.z, clip.w);
  output.uv = corner;
  output.color = surfel_norw[instanceIndex].xyz * 0.5 + 0.5;
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  // Round mask: uv is already in [-1,1] (corner space), discard outside the disc.
  if (length(input.uv) > 1.0) {
    discard;
  }
  return vec4f(input.color, 1.0);
}
    `,
);
