import { TColor } from '../../../../../../../../src/ECS/Components/Common.ts';

import { mutatedOptions, resetOptions, updateColorOptions } from '../Common/Options.ts';
import { createTankBase, createTankTurret } from '../Common/Tank.ts';
import {
    createTankCaterpillarsParts,
    createTankHullParts,
    createTankTurretParts,
    DENSITY,
    PADDING,
    PARTS_COUNT,
} from './LightTankParts.ts';

const TRACKS_COLOR = new Float32Array([0.6, 0.6, 0.6, 1]);
const TURRET_COLOR = new Float32Array([0.6, 1, 0.6, 1]);

export function createLightTank(options: {
    playerId: number,
    teamId: number,
    x: number,
    y: number,
    rotation: number,
    color: TColor,
}) {
    const tankOps = resetOptions(mutatedOptions, options);

    tankOps.density = DENSITY * 10;
    tankOps.width = PADDING * 12;
    tankOps.height = PADDING * 12;
    const [tankEid, tankPid] = createTankBase(PARTS_COUNT, tankOps);

    tankOps.density = DENSITY;
    tankOps.width = PADDING * 6;
    tankOps.height = PADDING * 6;
    const [turretEid] = createTankTurret(tankOps, tankEid, tankPid);

    createTankHullParts(tankEid, tankOps);

    {
        updateColorOptions(tankOps, TRACKS_COLOR);
        createTankCaterpillarsParts(tankEid, tankOps);
    }

    {
        updateColorOptions(tankOps, TURRET_COLOR);
        createTankTurretParts(turretEid, tankOps);
    }

    return tankEid;
}
