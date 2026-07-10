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
//   cone shader    : normalBias, aperture, giStrength, aoConeCount, aoReach, aoSteps,
//                    spNormalPow, spPlaneK, resolveRadius, ringR0/ringR1 (+ ringsOn)
//   composite shader: ambient, exposure, penumbra, shadowBaseSpread
//   screen-probe shader: conesPerProbe, maxDist (cone+probe reach), aperture, normalBias,
//                    temporalHysteresis, spNormalPow, spPlaneK, anisoMode
//   classify/decide/refine/debug shaders: ringR0/ringR1 (+ ringsOn); decide also lightThresh
//   cone-temporal shader: coneTemporalHysteresis
//   AIMED-cone group — emitterDirect, emitterFalloff, aimedSteps, aimedAlphaCut — bakes into the
//     screen-probe shader: the aimed emitter cones are traced once per PROBE (the probe-centric
//     final gather — see ./README.md).
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
  anisoMode: boolean; // far-field cone samples read the 6 anisotropic volumes (true) or the iso pyramid
  temporalHysteresis: number; // probe-atlas history weight (0..0.95; 0 = temporal OFF, fresh-only)
  // ── probe resolve (cone pass) + shared bilateral weights ─────────────────────────────
  spNormalPow: number; // normal-similarity sharpness (resolve taps + the gather's history validation)
  spPlaneK: number; // plane-reject threshold scale (× local probe spacing; × cellSize in the gather)
  resolveRadius: number; // resolve kernel support, in LOCAL probe pitches (bigger = smoother/wider)
  // ── adaptive density (decide/refine + foveated rings) ────────────────────────────────
  // Tile + divisor ALSO size the CPU-side atlas/buffers (screenProbeCounts) — the shaders bake
  // them as consts while rebuild() recreates the resources from the same config values, so the
  // two sides cannot drift. (maxAdaptive stays a live uniform: it derives from the CANVAS size.)
  screenProbeTile: number; // full-res px per uniform screen probe (the base lattice pitch)
  refineDiv: number; // refine cell divisor: an active tile subdivides into div² cells (cellPx = tile/div)
  lightThresh: number; // decide trigger: DC-luminance spread across the probe neighborhood
  ringsOn: boolean; // foveated rings master switch (off ⇒ level 0 everywhere — the flat lattice)
  ringR0: number; // rings: radial threshold where density drops to ÷4 (screen corner = 1)
  ringR1: number; // rings: radial threshold where density drops to ÷16
  // ── cone-output temporal filter ("point C") ──────────────────────────────────────────
  coneTemporalHysteresis: number; // history weight of the resolved-output blend (0 = passthrough)
};

// Effective ring thresholds: the OFF switch bakes as "thresholds past any on-screen radius"
// (corner = 1), so every shader keeps ONE code path and rings-off compiles to level 0 everywhere.
export function ringThresholds(cfg: VoxelBakedConfig): { r0: number; r1: number } {
  return cfg.ringsOn
    ? { r0: cfg.ringR0, r1: Math.max(cfg.ringR1, cfg.ringR0) }
    : { r0: 9, r1: 9 };
}

// Sanitized bake values for the probe lattice pitch + refine divisor — ONE clamp shared by every
// shader factory AND the CPU sizing (screenProbeCounts / dispatch math), so a fractional or zero
// config value cannot make the WGSL consts and the buffer strides disagree.
export function probeTile(cfg: VoxelBakedConfig): number {
  return Math.max(1, Math.round(cfg.screenProbeTile));
}
export function probeRefineDiv(cfg: VoxelBakedConfig): number {
  return Math.max(1, Math.round(cfg.refineDiv));
}

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
  anisoMode: true,
  temporalHysteresis: 0.75,
  spNormalPow: 2,
  spPlaneK: 1,
  resolveRadius: 2,
  screenProbeTile: 16, // Lumen default DownsampleFactor
  refineDiv: 2,
  lightThresh: 0.05,
  ringsOn: true,
  ringR0: 0.35,
  ringR1: 0.7,
  coneTemporalHysteresis: 0.85,
};
