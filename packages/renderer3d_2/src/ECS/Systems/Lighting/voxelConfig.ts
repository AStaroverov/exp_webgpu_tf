// BAKED voxel-GI configuration. These are quality / tuning knobs that are CONSTANT during
// gameplay (no per-frame animation), so instead of feeding them through per-frame uniform uploads
// they are "baked" as compile-time `const`s directly into the WGSL via the `wgsl` template tag
// (createConeShaderMeta / createCompositeShaderMeta / createProbeShaderMeta each interpolate these
// values). Changing one is an explicit, infrequent action: mutate the config and call
// voxelSystem.rebuild(), which recompiles the affected shaders + recreates their pipelines and
// bind groups with the new baked constants. Genuinely dynamic data (sun, camera invViewProj,
// emitter list, instance count, grid matrices, canvas size) stays in uniforms — it is NOT here.
//
// Field → where it bakes:
//   cone shader    : normalBias, aperture, giStrength, emitterDirect, emitterFalloff,
//                    aimedSteps, aimedAlphaCut, aoConeCount, aoReach, aoSteps
//   composite shader: ambient, exposure, penumbra
//   probe shader   : conesPerProbe, maxDist (cone+probe reach), aperture
//   probe-blur shader: probeBlurRadius
// (maxDist is unused by the cone shader body itself; it only drives the probe reach + CPU side.)
export type VoxelBakedConfig = {
  // ── cone pass ───────────────────────────────────────────────────────────────────────
  normalBias: number; // extra lift of the cone origin off the surface (world units)
  maxDist: number; // cone / probe reach (world units)
  aperture: number; // tan(halfAngle) — cone half-angle (~0.577 = 60° full angle)
  giStrength: number; // multiplier on the probe bounce (indirect) term
  emitterDirect: number; // multiplier on the summed emitter aimed-cone DIRECT light (vs the sun)
  emitterFalloff: number; // emitter distance falloff coefficient (0 = none/flat, 1 = standard 1/d²)
  aimedSteps: number; // aimed-cone march step budget (lower = cheaper, shorter/coarser shadows)
  aimedAlphaCut: number; // aimed-cone early-out opacity (<1 stops a near-opaque cone → saves the tail)
  aoConeCount: number; // short per-pixel hemisphere occlusion cones for contact AO (0 = no AO)
  aoReach: number; // AO cone reach (world units) — short, near-field contact occlusion
  aoSteps: number; // AO cone march budget (short)
  // ── composite pass ──────────────────────────────────────────────────────────────────
  ambient: number; // ambient floor (scaled by the cone's AO term)
  exposure: number; // HDR exposure multiplier applied before the ACES tonemap
  penumbra: number; // sun shadow softening strength: PCF widens as sun intensity drops below 1
  // ── probe pass ──────────────────────────────────────────────────────────────────────
  conesPerProbe: number; // full-sphere cones per probe; SH-L1 saturates ~16, so more only cuts noise
  // ── probe-blur pass ───────────────────────────────────────────────────────────────────
  probeBlurRadius: number; // 3D Gaussian blur radius (probes) over the SH volume; 0 = no blur.
  //   Spatially smooths the low-frequency bounce so a moving source's fill stops stepping by
  //   probe cells — far cheaper than raising probe resolution (O(probes·kernel), not ·cones).
};

export const DEFAULT_VOXEL_BAKED_CONFIG: VoxelBakedConfig = {
  normalBias: 0,
  maxDist: 24,
  aperture: 0.577,
  giStrength: 1,
  emitterDirect: 5,
  emitterFalloff: 1,
  aimedSteps: 32,
  aimedAlphaCut: 1,
  aoConeCount: 2,
  aoReach: 2,
  aoSteps: 12,
  ambient: 0.05,
  exposure: 1,
  penumbra: 4,
  conesPerProbe: 16,
  probeBlurRadius: 2,
};
