import { PI } from '../../../../../../../../lib/math.ts';
import { TColor } from '../../../../../../../renderer/src/ECS/Components/Common.ts';
import { BulletCaliber } from '../../../Components/Bullet.ts';
import { SlotPartType } from '../../../Components/SlotConfig.ts';
import { VehicleType } from '../../../Components/Vehicle.ts';
import { createSlotEntities, fillAllSlots } from '../../Vehicle/VehicleParts.ts';
import { getPhysicsOf } from '../../../refs.ts';
import { mutatedOptions, resetOptions, updateColorOptions } from '../Common/Options.ts';
import { createTankBase, createTankTracks, createTankTurret } from '../Common/Tank.ts';
import { createTankExhaustPipes } from '../../ExhaustPipe.ts';
import { Worlds } from '../../../../DI/Worlds.ts';
import {
    caterpillarLength,
    caterpillarSetLeft,
    caterpillarSetRight,
    DENSITY,
    hullSet,
    PADDING,
    PARTS_COUNT,
    SIZE,
    TRACK_ANCHOR_Y,
    turretGunSet,
    turretHeadSet,
} from './MediumTankParts.ts';
import { EngineType } from '../../../../Config/vehicles.ts';

const TRACKS_COLOR = new Float32Array([0.5, 0.5, 0.5, 1]);
const TURRET_COLOR = new Float32Array([0.5, 1, 0.5, 1]);
const APPROXIMATE_COLLIDER_RADIUS = 60;

export function createMediumTank(opts: {
    playerId: number,
    teamId: number,
    x: number,
    y: number,
    rotation: number,
    color: TColor,
}, { renderWorld } = Worlds) {
    const options = resetOptions(mutatedOptions, opts);
    options.partsCount = PARTS_COUNT;
    options.size = SIZE;
    options.padding = PADDING;
    options.approximateColliderRadius = APPROXIMATE_COLLIDER_RADIUS;
    options.vehicleType = VehicleType.MediumTank;
    options.engineType = EngineType.v8;
    options.trackLength = caterpillarLength;

    options.density = DENSITY * 14;
    options.width = PADDING * 11;
    options.height = PADDING * 7;
    const [tankPhysEid, tankRenderEid, tankPid] = createTankBase(options);

    // Create left and right tracks as independent entities
    const [leftTrackRenderEid, rightTrackRenderEid] = createTankTracks(
        options,
        {
            leftAnchorY: TRACK_ANCHOR_Y,
            rightAnchorY: -TRACK_ANCHOR_Y,
            anchorX: 0,
            trackWidth: PADDING * 2,
            trackHeight: caterpillarLength,
        },
        tankRenderEid,
        tankPid,
    );

    options.density = DENSITY;
    options.width = PADDING * 6;
    options.height = PADDING * 5;
    options.turret.rotationSpeed = PI * 0.7;
    options.turret.gunWidth = PADDING * 6;
    options.turret.gunHeight = PADDING * 2;
    options.firearms.bulletCaliber = BulletCaliber.Medium;
    options.firearms.reloadingDuration = 500;
    options.firearms.bulletStartPosition = [11 * PADDING, 0];
    const [turretRenderEid, gunRenderEid] = createTankTurret(tankRenderEid, tankPid, options);

    // Add exhaust pipes
    createTankExhaustPipes(renderWorld, tankRenderEid, PADDING * 11, PADDING * 7);

    // Carrier PHYSICS atoms (the EXACT atom owning each slot group).
    const leftTrackPhysEid = getPhysicsOf(leftTrackRenderEid);
    const rightTrackPhysEid = getPhysicsOf(rightTrackRenderEid);
    const turretPhysEid = getPhysicsOf(turretRenderEid);
    const gunPhysEid = getPhysicsOf(gunRenderEid);

    // Hull parts attached to tank body
    createSlotEntities(tankPhysEid, hullSet, options.color, SlotPartType.HullPart);

    // Caterpillar parts attached to track entities
    updateColorOptions(options, TRACKS_COLOR);
    createSlotEntities(leftTrackPhysEid, caterpillarSetLeft, options.color, SlotPartType.Caterpillar);
    createSlotEntities(rightTrackPhysEid, caterpillarSetRight, options.color, SlotPartType.Caterpillar);

    // Turret parts
    updateColorOptions(options, TURRET_COLOR);
    createSlotEntities(gunPhysEid, turretGunSet, options.color, SlotPartType.TurretGun);
    createSlotEntities(turretPhysEid, turretHeadSet, options.color, SlotPartType.TurretHead);

    // Fill all slots with physical parts
    fillAllSlots(tankPhysEid, options);
    fillAllSlots(leftTrackPhysEid, options);
    fillAllSlots(rightTrackPhysEid, options);
    fillAllSlots(turretPhysEid, options);
    fillAllSlots(gunPhysEid, options);

    return tankPhysEid;
}
