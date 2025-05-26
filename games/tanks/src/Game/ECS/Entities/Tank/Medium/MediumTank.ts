import { TColor } from '../../../../../../../../src/ECS/Components/Common.ts';
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
import { TankTurret } from '../../../Components/TankTurret.ts';
import { BulletCaliber } from '../../../Components/Bullet.ts';
import { Tank } from '../../../Components/Tank.ts';

const TRACKS_COLOR = new Float32Array([0.5, 0.5, 0.5, 1]);
const TURRET_COLOR = new Float32Array([0.5, 1, 0.5, 1]);

export function createMediumTank(opts: {
    playerId: number,
    teamId: number,
    x: number,
    y: number,
    rotation: number,
    color: TColor,
}) {
    const options = resetOptions(mutatedOptions, opts);
    options.size = SIZE;
    options.padding = PADDING;

    options.density = DENSITY * 10;
    options.width = PADDING * 14;
    options.height = PADDING * 14;
    const [tankEid, tankPid] = createTankBase(PARTS_COUNT, options);

    options.density = DENSITY;
    options.width = PADDING * 8;
    options.height = PADDING * 8;
    const [turretEid] = createTankTurret(options, tankEid, tankPid);


    {
        options.density = DENSITY * 10;
        createTankHullParts(tankEid, hullSet, options);
    }

    {
        options.density = DENSITY;
        updateColorOptions(options, TRACKS_COLOR);
        createTankCaterpillarsParts(tankEid, [caterpillarSetLeft, caterpillarSetRight], options);
        Tank.setCaterpillarsLength(tankEid, caterpillarLength);
    }

    {
        options.density = DENSITY;
        updateColorOptions(options, TURRET_COLOR);
        createTankTurretParts(turretEid, turretHeadSet, turretGunSet, options);
        TankTurret.setBulletData(turretEid, [0, -13 * PADDING], BulletCaliber.Medium);
        TankTurret.setReloadingDuration(turretEid, 200);
    }

    return tankEid;
}
