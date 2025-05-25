import { TColor } from '../../../../../../../../src/ECS/Components/Common.ts';
import {
    createTankCaterpillarsParts,
    createTankHullParts,
    createTankTurretParts,
    DENSITY,
    PADDING,
    PARTS_COUNT,
} from './MediumTankParts.ts';
import { mutatedOptions, resetOptions, updateColorOptions } from '../Common/Options.ts';
import { createTankBase, createTankTurret } from '../Common/Tank.ts';

const TRACKS_COLOR = new Float32Array([0.5, 0.5, 0.5, 1]);
const TURRET_COLOR = new Float32Array([0.5, 1, 0.5, 1]);

export function createMediumTank(options: {
    playerId: number,
    teamId: number,
    x: number,
    y: number,
    rotation: number,
    color: TColor,
}) {
    const tankOps = resetOptions(mutatedOptions, options);

    tankOps.density = DENSITY * 10;
    tankOps.width = PADDING * 14;
    tankOps.height = PADDING * 14;
    const [tankEid, tankPid] = createTankBase(PARTS_COUNT, tankOps);

    tankOps.density = DENSITY;
    tankOps.width = PADDING * 8;
    tankOps.height = PADDING * 8;
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
