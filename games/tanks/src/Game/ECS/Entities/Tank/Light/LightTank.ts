import { TColor } from '../../../../../../../../src/ECS/Components/Common.ts';

import { mutatedOptions, resetOptions, updateColorOptions } from '../Common/Options.ts';
import { createTankBase, createTankTurret } from '../Common/Tank.ts';
import {
    caterpillarLength,
    caterpillarSetLeft,
    caterpillarSetRight,
    DENSITY,
    hullSet,
    PADDING,
    PARTS_COUNT,
    turretGunSet,
    turretHeadSet,
} from './LightTankParts.ts';
import { BulletCaliber } from '../../../Components/Bullet.ts';
import { createTankCaterpillarsParts, createTankHullParts, createTankTurretParts } from '../Common/TankParts.ts';
import { SIZE } from '../Medium/MediumTankParts.ts';
import { PI } from '../../../../../../../../lib/math.ts';
import { TankEngineType } from '../../../Systems/Tank/TankControllerSystems.ts';

const TRACKS_COLOR = new Float32Array([0.6, 0.6, 0.6, 1]);
const TURRET_COLOR = new Float32Array([0.6, 1, 0.6, 1]);
const APPROXIMATE_COLLIDER_RADIUS = 50;

export function createLightTank(opts: {
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
    options.engineType = TankEngineType.v6;
    options.caterpillarLength = caterpillarLength;

    options.density = DENSITY * 10;
    options.width = PADDING * 12;
    options.height = PADDING * 12;
    const [tankEid, tankPid] = createTankBase(options);

    options.density = DENSITY;
    options.width = PADDING * 6;
    options.height = PADDING * 6;
    options.turret.rotationSpeed = PI * 0.8;
    options.turret.reloadingDuration = 100;
    options.turret.bulletCaliber = BulletCaliber.Light;
    options.turret.bulletStartPosition = [0, -9 * PADDING];
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
