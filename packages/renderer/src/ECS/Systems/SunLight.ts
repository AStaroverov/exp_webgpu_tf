/**
 * Single source of truth for the directional scene light (sun / moon / burning
 * city — whatever the scene calls for). Game code and the lighting GUI write
 * it; every consumer reads it directly each frame:
 *   - the RC lighting (escaped-ray sun lobe, packages/unknown Lighting),
 *   - the baked SDF z-shadows (SDFSystem/createDrawShapeSystem).
 */
export const SunLight = {
    enabled: true,
    /** Direction toward the sun, radians, screen frame (+X right, +Y down). */
    angle: 3.04,
};
