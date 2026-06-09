/**
 * Vehicle color palette.
 *
 * Team identity lives on the WEAPON: a vehicle's gun (the `TurretGun` slot — gun
 * barrel / launch rail) is always painted the team's base color, so you can tell
 * sides apart by the orudie regardless of the body.
 *
 * The body (hull / turret / tracks) uses one of several CONTRASTIVE palettes,
 * chosen at random per vehicle. Within a palette the three roles are deliberately
 * different colors (not shades of one hue), so the parts read as distinct; across
 * vehicles the random choice gives visual variety.
 */

export type VehiclePalette = {
    hull: Float32Array;
    turret: Float32Array;
    tracks: Float32Array;
};

const rgb = (r: number, g: number, b: number) => new Float32Array([r, g, b, 1]);

/**
 * Contrastive body palettes — each row is [hull, turret, tracks]. The three are
 * strongly separated in both hue and brightness so the parts read as clearly
 * distinct (bright hull, contrasting secondary, very dark tracks). Colors avoid
 * the pure team primaries (R/G/B/Y) so the team-colored turret+gun stands out
 * against the body. `turret` here is only used for the rocket tank's cabin —
 * standard tanks paint their turret with the team color.
 */
export const VEHICLE_PALETTES: VehiclePalette[] = [
    { hull: rgb(0.88, 0.74, 0.42), turret: rgb(0.16, 0.46, 0.48), tracks: rgb(0.11, 0.11, 0.12) }, // sand / teal
    { hull: rgb(0.80, 0.82, 0.85), turret: rgb(0.72, 0.36, 0.18), tracks: rgb(0.12, 0.12, 0.13) }, // light gray / rust
    { hull: rgb(0.58, 0.62, 0.26), turret: rgb(0.46, 0.30, 0.60), tracks: rgb(0.12, 0.12, 0.11) }, // olive / purple
    { hull: rgb(0.82, 0.64, 0.42), turret: rgb(0.28, 0.40, 0.64), tracks: rgb(0.12, 0.13, 0.14) }, // tan / slate-blue
    { hull: rgb(0.56, 0.82, 0.66), turret: rgb(0.48, 0.30, 0.18), tracks: rgb(0.10, 0.13, 0.11) }, // mint / brown
    { hull: rgb(0.94, 0.66, 0.50), turret: rgb(0.26, 0.28, 0.32), tracks: rgb(0.10, 0.10, 0.11) }, // peach / charcoal
    { hull: rgb(0.46, 0.62, 0.76), turret: rgb(0.92, 0.86, 0.64), tracks: rgb(0.11, 0.13, 0.15) }, // steel-blue / cream
    { hull: rgb(0.74, 0.68, 0.82), turret: rgb(0.20, 0.42, 0.26), tracks: rgb(0.12, 0.12, 0.13) }, // lavender / dark green
];

/** Pick a random contrastive body palette. */
export function randomVehiclePalette(): VehiclePalette {
    return VEHICLE_PALETTES[Math.floor(Math.random() * VEHICLE_PALETTES.length)];
}

/** Base identity color per team id — painted on the gun. */
export const TEAM_BASE_COLORS: Record<number, Float32Array> = {
    1: new Float32Array([0.90, 0.20, 0.20, 1]), // red
    2: new Float32Array([0.20, 0.45, 0.95, 1]), // blue
    3: new Float32Array([0.25, 0.80, 0.30, 1]), // green
    4: new Float32Array([0.95, 0.80, 0.15, 1]), // yellow
};

export const DEFAULT_TEAM_COLOR = TEAM_BASE_COLORS[1];

/** Team base color, falling back to team 1's color for unknown ids. */
export function teamBaseColor(teamId: number): Float32Array {
    return TEAM_BASE_COLORS[teamId] ?? DEFAULT_TEAM_COLOR;
}
