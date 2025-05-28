import { TColor } from '../../../../../../../renderer/src/ECS/Components/Common.ts';
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
} from './MediumTankParts.ts';
import { mutatedOptions, resetOptions, updateColorOptions } from '../Common/Options.ts';
import { createTankBase, createTankTurret } from '../Common/Tank.ts';
import { createTankCaterpillarsParts, createTankHullParts, createTankTurretParts } from '../Common/TankParts.ts';
import { BulletCaliber } from '../../../Components/Bullet.ts';
import { PI } from '../../../../../../../../lib/math.ts';
import { TankEngineType } from '../../../Systems/Tank/TankControllerSystems.ts';
import { TankType } from '../../../Components/Tank.ts';

const TRACKS_COLOR = new Float32Array([0.5, 0.5, 0.5, 1]);
const TURRET_COLOR = new Float32Array([0.5, 1, 0.5, 1]);
const APPROXIMATE_COLLIDER_RADIUS = 80;

export function createMediumTank(opts: {
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
    options.tankType = TankType.Medium;
    options.engineType = TankEngineType.v8;
    options.caterpillarLength = caterpillarLength;

    options.density = DENSITY * 10;
    options.width = PADDING * 14;
    options.height = PADDING * 14;
    const [tankEid, tankPid] = createTankBase(options);

    options.density = DENSITY;
    options.width = PADDING * 8;
    options.height = PADDING * 8;
    options.turret.rotationSpeed = PI * 0.6;
    options.turret.reloadingDuration = 200;
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
