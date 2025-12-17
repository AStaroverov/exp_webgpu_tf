import { TColor } from '../../../../../../../renderer/src/ECS/Components/Common.ts';

import { EntityId } from 'bitecs';
import { PI } from '../../../../../../../../lib/math.ts';
import { BulletCaliber } from '../../../Components/Bullet.ts';
import { SlotPartType } from '../../../Components/SlotConfig.ts';
import { VehicleType } from '../../../Components/Vehicle.ts';
import { VehicleEngineType } from '../../../Systems/Vehicle/VehicleControllerSystems.ts';
import { createSlotEntities, fillAllSlots, updateSlotsBrightness } from '../../Vehicle/VehicleParts.ts';
import { mutatedOptions, resetOptions, updateColorOptions } from '../Common/Options.ts';
import { createTankBase, createTankTurret } from '../Common/Tank.ts';
import { SIZE } from '../Medium/MediumTankParts.ts';
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
}): EntityId {
    const options = resetOptions(mutatedOptions, opts);
    options.partsCount = PARTS_COUNT;
    options.size = SIZE;
    options.padding = PADDING;
    options.approximateColliderRadius = APPROXIMATE_COLLIDER_RADIUS;
    options.vehicleType = VehicleType.LightTank;
    options.engineType = VehicleEngineType.v6;
    options.caterpillarLength = caterpillarLength;

    options.density = DENSITY * 10;
    options.width = PADDING * 12;
    options.height = PADDING * 12;
    const [tankEid, tankPid] = createTankBase(options);

    options.density = DENSITY;
    options.width = PADDING * 6;
    options.height = PADDING * 6;
    options.turret.rotationSpeed = PI * 0.8;
    options.firearms.reloadingDuration = 300;
    options.firearms.bulletCaliber = BulletCaliber.Light;
    options.firearms.bulletStartPosition = [0, -9 * PADDING];
    const [turretEid] = createTankTurret(options, tankEid, tankPid);

    createSlotEntities(tankEid, hullSet, options.color, SlotPartType.HullPart);

    updateColorOptions(options, TRACKS_COLOR);
    createSlotEntities(tankEid, caterpillarSetLeft, options.color, SlotPartType.Caterpillar);
    createSlotEntities(tankEid, caterpillarSetRight, options.color, SlotPartType.Caterpillar);

    updateColorOptions(options, TURRET_COLOR);
    createSlotEntities(turretEid, turretGunSet, options.color, SlotPartType.TurretGun);
    createSlotEntities(turretEid, turretHeadSet, options.color, SlotPartType.TurretHead);

    updateSlotsBrightness(tankEid);
    fillAllSlots(tankEid, options);
    updateSlotsBrightness(turretEid);
    fillAllSlots(turretEid, options);

    return tankEid;
}
