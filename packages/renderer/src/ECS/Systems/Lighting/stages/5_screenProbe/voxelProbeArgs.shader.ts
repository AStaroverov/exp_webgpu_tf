import { VariableKind, VariableMeta } from "../../../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../../../WGSL/wgsl.ts";
import { GATHER_WORKGROUP } from "./voxelScreenProbe.shader.ts";

// Adaptive screen-probe atlas — PASS B (build indirect args). A single thread reads the adaptive
// probe count the refine pass allocated (uCounter[0], capped at maxAdaptive) and writes the ADAPTIVE
// gather's dispatchWorkgroupsIndirect args = [ceil(adaptive / GATHER_WG), 1, 1], so the adaptive
// gather launches EXACTLY over the adaptiveCount probes with no wasted groups on the empty atlas
// tail. The uniform gather is a separate DIRECT dispatch (its size, numUniform, is CPU-known), so
// only the adaptive count needs an on-GPU args build (the count is atomic → known only on the GPU).
//
// group 0 = uParams uniform (numUniform + maxAdaptive). group 1 = uCounter (StorageRead: the atomic
// buffer read back as plain u32 now that the refine pass's writes are visible across the barrier).
// group 2 = uArgs (StorageWrite) — bound to the RAW STORAGE|INDIRECT buffer, exactly the smoke
// test's argsMeta shape.

export const argsShaderMeta = new ShaderMeta(
  {
    // .x = numUniform, .y = maxAdaptive, .z/.w spare.
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<u32>`, {
      visibility: GPUShaderStage.COMPUTE,
    }),
    // The allocator counter, read as plain u32 (non-atomic) — safe: PASS B runs after PASS A1 in a
    // separate compute pass, so the value is final.
    counter: new VariableMeta("uCounter", VariableKind.StorageRead, `array<u32, 2>`, {
      visibility: GPUShaderStage.COMPUTE,
    }),
    args: new VariableMeta("uArgs", VariableKind.StorageWrite, `array<u32, 3>`, {
      visibility: GPUShaderStage.COMPUTE,
    }),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
@compute @workgroup_size(1)
fn main() {
  // uParams.x (numUniform) is now the uniform gather's DIRECT dispatch size (CPU-known) — not used
  // here; only the adaptive count drives the indirect args. Kept declared/bound for a stable layout.
  let adaptive = min(uCounter[0], uParams.y);      // cap at the budget (overflow was not placed)
  uArgs[0] = (adaptive + ${GATHER_WORKGROUP}u - 1u) / ${GATHER_WORKGROUP}u;  // ceil(adaptive / GATHER_WG)
  uArgs[1] = 1u;
  uArgs[2] = 1u;
}
`,
);
