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
//                    screenProbeTile, spNormalPow, spPlaneK, resolveRadius
//   composite shader: ambient, exposure, penumbra, shadowBaseSpread
//   screen-probe shader: conesPerProbe, maxDist (cone+probe reach), aperture, normalBias,
//                    screenProbeTile, temporalHysteresis, spNormalPow, spPlaneK, anisoMode
//   probe-debug shader: screenProbeTile
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
  // The tile ALSO sizes the CPU-side probe atlas — the shaders bake it as a const while rebuild()
  // recreates the textures from the same config value, so the two sides cannot drift.
  screenProbeTile: number; // full-res px per screen probe (the lattice pitch)
  // ── cone-output temporal filter ("point C") ──────────────────────────────────────────
  coneTemporalHysteresis: number; // history weight of the resolved-output blend (0 = passthrough)
};

// Sanitized bake value for the probe lattice pitch — ONE clamp shared by every shader factory AND
// the CPU sizing (atlas dims / dispatch math), so a fractional or zero config value cannot make
// the WGSL const and the texture dims disagree.
export function probeTile(cfg: VoxelBakedConfig): number {
  return Math.max(1, Math.round(cfg.screenProbeTile));
}

// ===== GI QUALITY PRESETS (low / medium / high). =====
// One switch over the PERF-relevant baked knobs: probe density (tile), cone budgets (fill / aimed /
// AO), reach, the aniso anti-leak, the two temporal hystereses (a cheaper preset leans HARDER on
// temporal accumulation to hide its noise), and the resolve kernel width (sparser probes want a
// wider kernel). Artistic tuning (giStrength, exposure, ambient, sun/emitter strengths, aperture…)
// is deliberately NOT in the presets — switching quality must never change the LOOK the user dialed
// in, only its fidelity/cost.
//
// Every preset overrides the SAME key set, so switching is deterministic in any order (no key
// leaks from a previous preset). `medium` == DEFAULT_VOXEL_BAKED_CONFIG for these keys.
// coneScale is the one non-baked lever (the cone target's downscale — a texture size, not a shader
// const); it rides the preset so the biggest perf lever isn't left behind.
// Apply: Object.assign(config, preset.config) + setConeScale(preset.coneScale) + rebuild().
export type GIQuality = "low" | "medium" | "high";
export type GIQualityPreset = {
  config: Partial<VoxelBakedConfig>;
  coneScale: number;
};
export const GI_QUALITY_PRESETS: Record<GIQuality, GIQualityPreset> = {
  // ~2–3× cheaper than medium, SAME probe density and resolve resolution (dropping those is what
  // wrecks the image — validated in-browser). Savings come from the trace budgets only: half the
  // fill cones, shorter reach, half the aimed/AO work, iso-only far field; hysteresis raised so
  // temporal accumulation integrates the thinner per-frame sampling.
  low: {
    config: {
      screenProbeTile: 16,
      conesPerProbe: 8,
      maxDist: 16,
      aimedSteps: 12,
      aimedPerFrame: 4,
      aoConeCount: 1,
      aoSteps: 8,
      anisoMode: false,
      temporalHysteresis: 0.85,
      coneTemporalHysteresis: 0.9,
      resolveRadius: 2,
    },
    coneScale: 2,
  },
  // The defaults (the shipped baseline).
  medium: {
    config: {
      screenProbeTile: 16,
      conesPerProbe: 16,
      maxDist: 24,
      aimedSteps: 16,
      aimedPerFrame: 8,
      aoConeCount: 2,
      aoSteps: 12,
      anisoMode: true,
      temporalHysteresis: 0.75,
      coneTemporalHysteresis: 0.85,
      resolveRadius: 2,
    },
    coneScale: 2,
  },
  // ×4 probes (tile 8), double budgets everywhere, longer reach, tighter kernel; hysteresis
  // lowered — dense fresh sampling needs less temporal smoothing, so light reacts faster.
  high: {
    config: {
      screenProbeTile: 8,
      conesPerProbe: 32,
      maxDist: 32,
      aimedSteps: 24,
      aimedPerFrame: 12,
      aoConeCount: 4,
      aoSteps: 16,
      anisoMode: true,
      temporalHysteresis: 0.7,
      coneTemporalHysteresis: 0.8,
      resolveRadius: 1.5,
    },
    coneScale: 2,
  },
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
  anisoMode: true,
  temporalHysteresis: 0.75,
  spNormalPow: 2,
  spPlaneK: 1,
  resolveRadius: 2,
  screenProbeTile: 16, // Lumen default DownsampleFactor
  coneTemporalHysteresis: 0.85,
};
