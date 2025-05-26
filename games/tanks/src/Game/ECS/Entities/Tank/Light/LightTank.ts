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
import { Tank } from '../../../Components/Tank.ts';
import { TankTurret } from '../../../Components/TankTurret.ts';
import { BulletCaliber } from '../../../Components/Bullet.ts';
import { createTankCaterpillarsParts, createTankHullParts, createTankTurretParts } from '../Common/TankParts.ts';

const TRACKS_COLOR = new Float32Array([0.6, 0.6, 0.6, 1]);
const TURRET_COLOR = new Float32Array([0.6, 1, 0.6, 1]);

export function createLightTank(opts: {
    playerId: number,
    teamId: number,
    x: number,
    y: number,
    rotation: number,
    color: TColor,
}) {
    const options = resetOptions(mutatedOptions, opts);

    options.density = DENSITY * 10;
    options.width = PADDING * 12;
    options.height = PADDING * 12;
    const [tankEid, tankPid] = createTankBase(PARTS_COUNT, options);

    options.density = DENSITY;
    options.width = PADDING * 6;
    options.height = PADDING * 6;
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
        TankTurret.setBulletData(turretEid, [0, -9 * PADDING], BulletCaliber.Light);
        TankTurret.setReloadingDuration(turretEid, 100);
    }

    return tankEid;
}
