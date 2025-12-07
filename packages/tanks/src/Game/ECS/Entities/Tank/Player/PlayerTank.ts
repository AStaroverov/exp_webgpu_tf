import { PI } from '../../../../../../../../lib/math.ts';
import { TColor } from '../../../../../../../renderer/src/ECS/Components/Common.ts';
import { BulletCaliber } from '../../../Components/Bullet.ts';
import { TankType } from '../../../Components/Tank.ts';
import { TankEngineType } from '../../../Systems/Tank/TankControllerSystems.ts';
import { mutatedOptions, resetOptions, updateColorOptions } from '../Common/Options.ts';
import { createTankBase, createTankTurret } from '../Common/Tank.ts';
import { createTankCaterpillarsParts, createTankHullParts, createTankTurretParts } from '../Common/TankParts.ts';
// Use Medium tank parts as base
import {
    caterpillarLength,
    caterpillarSetLeft,
    caterpillarSetRight,
    DENSITY,
    hullSet,
    PADDING,
    PARTS_COUNT,
    SIZE,
    turretGunSet,
    turretHeadSet,
} from '../Medium/MediumTankParts.ts';

const TRACKS_COLOR = new Float32Array([0.4, 0.4, 0.5, 1]);
const TURRET_COLOR = new Float32Array([0.4, 0.8, 1, 1]);  // Bluish tint for player
const APPROXIMATE_COLLIDER_RADIUS = 80;

/**
 * Player Tank - based on Medium tank but with:
 * - v8_turbo engine (40% faster movement, 50% faster rotation)
 * - Faster reload (350ms vs 500ms)
 * - Faster turret rotation
 */
export function createPlayerTank(opts: {
    playerId: number,
    teamId: number,
    x: number,
    y: number,
    rotation: number,
    color: TColor,
}) {
    const options = resetOptions(mutatedOptions, opts);
    options.partsCount = PARTS_COUNT;
    options.size = SIZE;
    options.padding = PADDING;
    options.approximateColliderRadius = APPROXIMATE_COLLIDER_RADIUS;
    options.tankType = TankType.Player;
    options.engineType = TankEngineType.v8_turbo;  // Faster engine
    options.caterpillarLength = caterpillarLength;

    options.density = DENSITY * 10;
    options.width = PADDING * 14;
    options.height = PADDING * 14;
    const [tankEid, tankPid] = createTankBase(options);

    options.density = DENSITY;
    options.width = PADDING * 8;
    options.height = PADDING * 8;
    options.turret.rotationSpeed = PI * 0.8;        // Faster turret (was 0.6)
    options.turret.reloadingDuration = 350;          // Faster reload (was 500)
    options.turret.bulletCaliber = BulletCaliber.Medium;
    options.turret.bulletStartPosition = [0, -13 * PADDING];
    const [turretEid] = createTankTurret(options, tankEid, tankPid);

    {
        options.density = DENSITY * 10;
        createTankHullParts(tankEid, hullSet, options);
    }

    {
        options.density = DENSITY;
        updateColorOptions(options, TRACKS_COLOR);
        createTankCaterpillarsParts(tankEid, [caterpillarSetLeft, caterpillarSetRight], options);
    }

    {
        options.density = DENSITY;
        updateColorOptions(options, TURRET_COLOR);
        createTankTurretParts(turretEid, turretHeadSet, turretGunSet, options);
    }

    return tankEid;
}
