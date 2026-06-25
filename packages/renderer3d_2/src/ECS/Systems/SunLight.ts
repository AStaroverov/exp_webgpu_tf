/**
 * Single source of truth for the directional scene light (sun / moon / whatever the
 * scene calls for). Game code and the lighting GUI write it; every consumer reads it
 * directly each frame. World-space directional light, Z-up frame.
 *
 * The direction TOWARD the sun is built from two angles:
 *   dir = (cos(angle)·cos(elevation), sin(angle)·cos(elevation), sin(elevation))
 * where `angle` is the azimuth in the world XY plane (radians) and `elevation` is the
 * angle above that plane (radians).
 */
export const SunLight = {
  enabled: true,
  /** Azimuth in the world XY plane, radians. */
  angle: 3.04,
  /** Angle above the XY plane, radians. */
  elevation: 0.9,
  /** Scalar light intensity. */
  intensity: 1.5,
  /** Linear RGB color, components in 0..1. */
  color: [1.0, 0.95, 0.85] as [number, number, number],
};
