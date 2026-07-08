// BAKED voxel-GI configuration. These are quality / tuning knobs that are CONSTANT during
// gameplay (no per-frame animation), so instead of feeding them through per-frame uniform uploads
// they are "baked" as compile-time `const`s directly into the WGSL via the `wgsl` template tag
// (createConeShaderMeta / createCompositeShaderMeta / createScreenProbeShaderMeta each interpolate these
// values). Changing one is an explicit, infrequent action: mutate the config and call
// voxelSystem.rebuild(), which recompiles the affected shaders + recreates their pipelines and
// bind groups with the new baked constants. Genuinely dynamic data (sun, camera invViewProj,
// emitter list, instance count, grid matrices, canvas size) stays in uniforms — it is NOT here.
//
// Field → where it bakes:
//   cone shader    : normalBias, aperture, giStrength, aoConeCount, aoReach, aoSteps
//   composite shader: ambient, exposure, penumbra, shadowBaseSpread
//   screen-probe shader: conesPerProbe, maxDist (cone+probe reach), aperture, normalBias
//   AIMED-cone group — emitterDirect, emitterFalloff, aimedSteps, aimedAlphaCut — bakes into the
//     screen-probe shader: the aimed emitter cones are traced once per PROBE (the probe-centric
//     final gather — see docs/probe-centric-gi-migration.md).
// (maxDist is unused by the cone shader body itself; it only drives the probe reach + CPU side.)
export type VoxelBakedConfig = {
  // ── cone pass ───────────────────────────────────────────────────────────────────────
  normalBias: number; // extra lift of the cone origin off the surface (world units)
  maxDist: number; // cone / probe reach (world units)
  aperture: number; // tan(halfAngle) — cone half-angle (~0.577 = 60° full angle)
  giStrength: number; // multiplier on the probe bounce (indirect) term
  // The aimed-cone group below bakes into the screen-probe shader (see the field→shader map above).
  emitterDirect: number; // multiplier on the summed emitter aimed-cone DIRECT light (vs the sun)
  emitterFalloff: number; // emitter distance falloff coefficient (0 = none/flat, 1 = standard 1/d²)
  aimedSteps: number; // aimed-cone march step budget (lower = cheaper, shorter/coarser shadows)
  aimedAlphaCut: number; // aimed-cone early-out opacity (<1 stops a near-opaque cone → saves the tail)
  aimedPerFrame: number; // aimed cones traced per probe per FRAME; with more live lights each probe
  //   round-robins a window of this many (energy-rescaled, temporally integrated) → the aimed cost
  //   is CONSTANT in the light count. Requires temporal hysteresis > 0 once lights exceed it.
  clusterDiv: number; // light-cluster cell size in VOXELS per axis (clustered light culling à la
  //   Persson: CPU bins each emitter into the world-space cells its influence sphere overlaps;
  //   a probe round-robins ONLY its own cell's list). Smaller = tighter lists, more CPU binning.
  clusterCap: number; // max lights recorded per cluster cell (overflow lights are dropped for that
  //   cell — raise it if a scene legitimately packs more overlapping emitters than this).
  aoConeCount: number; // short per-pixel hemisphere occlusion cones for contact AO (0 = no AO)
  aoReach: number; // AO cone reach (world units) — short, near-field contact occlusion
  aoSteps: number; // AO cone march budget (short)
  // ── composite pass ──────────────────────────────────────────────────────────────────
  ambient: number; // ambient floor (scaled by the cone's AO term)
  exposure: number; // HDR exposure multiplier applied before the ACES tonemap
  penumbra: number; // sun shadow softening strength: PCF widens as sun intensity drops below 1
  shadowBaseSpread: number; // base sun-shadow PCF radius (texels) ALWAYS applied, even at full sun;
  //   smooths the shadow-map texel staircase into a soft edge. 1 = near-hard (old behavior).
  // ── screen-probe pass ─────────────────────────────────────────────────────────────────
  conesPerProbe: number; // full-sphere cones per screen probe; SH-L1 saturates ~16, so more only cuts noise
};

export const DEFAULT_VOXEL_BAKED_CONFIG: VoxelBakedConfig = {
  normalBias: 0,
  maxDist: 24,
  aperture: 0.577,
  giStrength: 1,
  emitterDirect: 2,
  emitterFalloff: 1,
  aimedSteps: 16,
  aimedAlphaCut: 1,
  aimedPerFrame: 8,
  clusterDiv: 8,
  clusterCap: 16,
  aoConeCount: 2,
  aoReach: 2,
  aoSteps: 12,
  ambient: 0.05,
  exposure: 1,
  penumbra: 4,
  shadowBaseSpread: 2,
  conesPerProbe: 16,
};
