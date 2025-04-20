import { ZIndex } from '../../../consts.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { TColor } from '../../../../../../src/ECS/Components/Common.ts';
import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { GameDI } from '../../../DI/GameDI.ts';
import { RigidBodyRef } from '../../Components/Physical.ts';
import { createRectangleRR } from '../../Components/RigidRender.ts';
import { PlayerRef } from '../../Components/PlayerRef.ts';
import { Hitable } from '../../Components/Hitable.ts';
import { Parent } from '../../Components/Parent.ts';
import { Children } from '../../Components/Children.ts';
import { TankPart, TankPartTrack } from '../../Components/TankPart.ts';
import { BASE_DENSITY, Options, updateColorOptions } from './Options.ts';
import { randomRangeFloat } from '../../../../../../lib/random.ts';
import { clamp } from 'lodash-es';
import { addComponent } from 'bitecs';
import { TeamRef } from '../../Components/TeamRef.ts';

export const SIZE = 5;
export const PADDING = SIZE + 1;
const hullSet: [number, number, number, number][] =
    Array.from({ length: 88 }, (_, i) => {
        return [
            i * PADDING % (PADDING * 8), Math.floor(i / 8) * PADDING,
            SIZE, SIZE,
        ];
    });
const turretSet: [number, number, number, number][] =
    Array.from({ length: 42 }, (_, i): [number, number, number, number] => {
        return [
            -PADDING * 2 + i * PADDING % (PADDING * 6), 10 * PADDING + Math.floor(i / 6) * PADDING,
            SIZE, SIZE,
        ];
    });
const gunSet: [number, number, number, number][] =
    Array.from({ length: 20 }, (_, i): [number, number, number, number] => {
        return [
            i * PADDING % (PADDING * 2), Math.floor(i / 2) * PADDING,
            SIZE, SIZE,
        ];
    });

export const CATERPILLAR_SIZE = 3;
export const CATERPILLAR_PADDING = CATERPILLAR_SIZE + 1;
export const CATERPILLAR_LINE_COUNT = 18;
const caterpillarSet: [number, number, number, number][] =
    Array.from({ length: CATERPILLAR_LINE_COUNT * 2 }, (_, i) => {
        return [
            i * SIZE % (SIZE * 2), Math.floor(i / 2) * CATERPILLAR_PADDING,
            SIZE, CATERPILLAR_SIZE,
        ];
    });

export const CATERPILLAR_COUNT = CATERPILLAR_LINE_COUNT * 2;
export const PARTS_COUNT = hullSet.length + turretSet.length + gunSet.length + CATERPILLAR_COUNT * 2;

/**
 * Создает корпус танка
 */
export function createTankHullParts(options: Options, tankEid: number, color: TColor): void {
    options.z = ZIndex.TankHull;
    options.density = BASE_DENSITY * 10;
    options.belongsSolverGroup = CollisionGroup.ALL;
    options.interactsSolverGroup = CollisionGroup.ALL;
    options.belongsCollisionGroup = CollisionGroup.TANK_HULL_PARTS;
    options.interactsCollisionGroup = CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_HULL_PARTS;

    updateColorOptions(options, color);
    createThingFromParts(options, {
        parentEId: tankEid,
        params: hullSet,
        x: 0 - 3.5 * PADDING,
        y: 0 - 5 * PADDING,
    });
}

const tracksColor = new Float32Array([0.5, 0.5, 0.5, 1]);

export function createTankTracksParts(options: Options, tankEid: number): void {
    updateColorOptions(options, tracksColor);
    createThingFromParts(options, {
        parentEId: tankEid,
        params: caterpillarSet,
        tag: TankPartTrack,
        x: -5.7 * PADDING,
        y: -6 * PADDING,
    });
    updateColorOptions(options, tracksColor);
    createThingFromParts(options, {
        parentEId: tankEid,
        params: caterpillarSet,
        tag: TankPartTrack,
        x: 4.7 * PADDING,
        y: -6 * PADDING,
    });
}


const turretColor = new Float32Array([0.5, 1, 0.5, 1]);

export function createTankTurretAndGunParts(options: Options, turretEid: number) {
    options.z = ZIndex.TankTurret;
    options.density = BASE_DENSITY;
    options.belongsCollisionGroup = CollisionGroup.TANK_TURRET_PARTS;
    options.interactsCollisionGroup = CollisionGroup.ALL | CollisionGroup.WALL | CollisionGroup.TANK_TURRET_PARTS | CollisionGroup.TANK_GUN_PARTS;

    updateColorOptions(options, turretColor);
    createThingFromParts(options, {
        parentEId: turretEid,
        params: turretSet,
        x: -0.5 * PADDING,
        y: -8 * PADDING,
    });

    options.shadow[1] = 4;
    options.belongsCollisionGroup = CollisionGroup.TANK_GUN_PARTS;
    options.interactsCollisionGroup = CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_TURRET_PARTS | CollisionGroup.TANK_GUN_PARTS;

    updateColorOptions(options, turretColor);
    createThingFromParts(options, {
        parentEId: turretEid,
        params: gunSet,
        x: -0.5 * PADDING,
        y: -8 * PADDING,
    });
}

const parentVector = new Vector2(0, 0);
const childVector = new Vector2(0, 0);

function createThingFromParts(
    options: Options,
    {
        parentEId,
        params,
        tag,
        x,
        y,
    }: {
        parentEId: number,
        params: [number, number, number, number][],
        tag?: object
        x: number,
        y: number,
    },
    { world, physicalWorld } = GameDI,
) {
    const rbId = RigidBodyRef.id[parentEId];
    const parentRb = physicalWorld.getRigidBody(rbId);

    childVector.x = 0;
    childVector.y = 0;

    const baseColor = new Float32Array(options.color);

    return params.map((param, i) => {
        const parentTranslation = parentRb.translation();
        options.x = parentTranslation.x + x + param[0];
        options.y = parentTranslation.y + y + param[1];
        options.width = param[2];
        options.height = param[3];
        options.color = new Float32Array(baseColor);
        adjustBrightness(options.color, i / params.length / 3 - 0.1, i / params.length / 3 + 0.1);

        const [eid, pid] = createRectangleRR(options);

        parentVector.x = x + param[0];
        parentVector.y = y + param[1];
        const joint = physicalWorld.createImpulseJoint(
            JointData.fixed(parentVector, 0, childVector, 0),
            parentRb,
            physicalWorld.getRigidBody(pid),
            true,
        );
        TankPart.addComponent(world, eid, joint.handle, parentVector, childVector);

        PlayerRef.addComponent(world, eid, options.playerId);
        TeamRef.addComponent(world, eid, options.teamId);
        Hitable.addComponent(world, eid);
        Parent.addComponent(world, eid, parentEId);
        Children.addChildren(parentEId, eid);

        tag && addComponent(world, eid, tag);

        return eid;
    });
}

function adjustBrightness(color: TColor, start: number, end: number) {
    const factor = -1 * randomRangeFloat(start, end);
    color[0] = clamp(color[0] + factor, 0, 1);
    color[1] = clamp(color[1] + factor, 0, 1);
    color[2] = clamp(color[2] + factor, 0, 1);
}