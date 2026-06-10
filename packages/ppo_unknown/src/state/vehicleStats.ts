/**
 * vehicleStats — the SINGLE source of truth for the units vector's per-unit
 * attribute features (`role, mobility, firepower, reload, range`), normalized ONCE
 * from the REAL `unknown` game configs (no magic numbers float around the snapshot).
 *
 * All five are 0..1. Ratios are taken straight from the configs so they track any
 * config change automatically:
 *   - mobility   ← `EngineConfig[engine].impulseMult`, divided by the fastest engine.
 *   - firepower  ← `BulletCaliberConfig[caliber].damage`, divided by the hardest hit.
 *   - reload     ← `reloadTime`, INVERTED (faster reload = higher value):
 *                  `minReload / reloadTime`, so the quickest gun reads 1.0.
 *   - range      ← `BulletCaliberConfig[caliber].maxDistance` (bullet flight
 *                  distance), divided by the longest range. Same source the
 *                  predicted-fire projection in `markBulletThreat` uses.
 *   - role       ← every tank is a gun fighter: `RoleStat.Fighter`.
 */

import {
    EngineConfig,
    getTankConfig,
    VehicleType,
} from '../../../unknown/src/Game/Config/vehicles.ts';
import { BulletCaliberConfig } from '../../../unknown/src/Game/Config/weapons.ts';

/** Coarse unit role, normalized into a single 0..1 feature. */
export const RoleStat = {
    /** Gun vehicle (Light/Medium/Heavy tank). */
    Fighter: 0,
} as const;

/** Normalized observation attributes for one vehicle type. All fields 0..1. */
export type VehicleStats = {
    /** `RoleStat`: 0 = fighter (gun). */
    role: number;
    /** Engine impulse relative to the fastest engine. */
    mobility: number;
    /** Bullet damage relative to the hardest-hitting gun; 0 if gunless. */
    firepower: number;
    /** Reload speed (INVERTED time) relative to the quickest gun; 0 if gunless. */
    reload: number;
    /** Bullet flight distance relative to the longest range; 0 if gunless. */
    range: number;
};

/** The gun vehicle types whose stats define the normalizers. */
const GUN_TYPES = [VehicleType.LightTank, VehicleType.MediumTank, VehicleType.HeavyTank];

/** Fastest engine impulse across ALL engines (the mobility normalizer). */
const MAX_IMPULSE = Math.max(
    ...Object.values(EngineConfig).map((e) => e.impulseMult),
);

/** Hardest-hitting gun damage (the firepower normalizer). */
const MAX_DAMAGE = Math.max(
    ...GUN_TYPES.map((t) => BulletCaliberConfig[getTankConfig(t).gun!.caliber].damage),
);

/** Quickest reload time across guns (the inverted-reload numerator). */
const MIN_RELOAD = Math.min(
    ...GUN_TYPES.map((t) => BulletCaliberConfig[getTankConfig(t).gun!.caliber].reloadTime),
);

/** Longest bullet flight distance (the range normalizer). */
const MAX_RANGE = Math.max(
    ...GUN_TYPES.map((t) => BulletCaliberConfig[getTankConfig(t).gun!.caliber].maxDistance),
);

function computeStats(type: VehicleType): VehicleStats {
    const config = getTankConfig(type);
    const mobility = EngineConfig[config.engine].impulseMult / MAX_IMPULSE;
    const gun = config.gun;
    const caliber = gun ? BulletCaliberConfig[gun.caliber] : null;
    return {
        role: RoleStat.Fighter,
        mobility,
        reload: caliber ? MIN_RELOAD / caliber.reloadTime : 0,
        range: caliber ? caliber.maxDistance / MAX_RANGE : 0,
        firepower: caliber ? caliber.damage / MAX_DAMAGE : 0,
    };
}

/** Every spawnable tank type (gun and stream; non-tank types have no stats row). */
const TANK_STAT_TYPES = [
    VehicleType.LightTank,
    VehicleType.MediumTank,
    VehicleType.HeavyTank,
    VehicleType.RocketTank,
    VehicleType.FlameTank,
    VehicleType.FrostTank,
];

/** Precomputed normalized stats per tank `VehicleType` (computed once at module load). */
const STATS: readonly VehicleStats[] = TANK_STAT_TYPES.reduce<VehicleStats[]>((acc, t) => {
    acc[t] = computeStats(t);
    return acc;
}, []);

/** Normalized observation stats for a vehicle type. Single source of truth. */
export function getVehicleStats(type: VehicleType): VehicleStats {
    return STATS[type];
}
