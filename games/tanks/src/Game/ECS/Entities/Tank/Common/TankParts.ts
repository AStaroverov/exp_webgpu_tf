import { ZIndex } from '../../../../consts.ts';
import { CollisionGroup } from '../../../../Physical/createRigid.ts';
import { TColor } from '../../../../../../../../src/ECS/Components/Common.ts';
import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { GameDI } from '../../../../DI/GameDI.ts';
import { RigidBodyRef } from '../../../Components/Physical.ts';
import { createRectangleRR } from '../../../Components/RigidRender.ts';
import { PlayerRef } from '../../../Components/PlayerRef.ts';
import { Hitable } from '../../../Components/Hitable.ts';
import { Parent } from '../../../Components/Parent.ts';
import { Children } from '../../../Components/Children.ts';
import { TankPart, TankPartCaterpillar } from '../../../Components/TankPart.ts';
import { Options } from './Options.ts';
import { randomRangeFloat } from '../../../../../../../../lib/random.ts';
import { clamp } from 'lodash-es';
import { addComponent } from 'bitecs';
import { TeamRef } from '../../../Components/TeamRef.ts';
import { Tank } from '../../../Components/Tank.ts';
import { TankTurret } from '../../../Components/TankTurret.ts';
import { BulletCaliber } from '../../../Components/Bullet.ts';

export type PartsData = [x: number, y: number, w: number, h: number];

export function createRectangleSet(
    cols: number, rows: number,
    width: number, paddingWidth: number,
    height = width, paddingHeight = paddingWidth,
): PartsData[] {
    const count = cols * rows;
    return Array.from({ length: count }, (_, i) => {
        return [
            i * paddingWidth % (paddingWidth * cols) - (paddingWidth * cols / 2 - width / 2),
            Math.floor(i / cols) * paddingHeight - (paddingHeight * rows / 2 - height / 2),
            width, height,
        ];
    });
}

export function createTankHullParts(tankEid: number, hullSet: PartsData[], options: Options): void {
    options.z = ZIndex.TankHull;
    options.belongsSolverGroup = CollisionGroup.ALL;
    options.interactsSolverGroup = CollisionGroup.ALL;
    options.belongsCollisionGroup = CollisionGroup.TANK_HULL_PARTS;
    options.interactsCollisionGroup = CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_HULL_PARTS;

    createThingFromParts({
        parentEId: tankEid,
        params: hullSet,
    }, options);
}

export function createTankCaterpillarsParts(tankEid: number, caterpillars: [lineCount: number, PartsData[]][], options: Options): void {
    options.z = ZIndex.TankCaterpillar;
    options.belongsSolverGroup = CollisionGroup.ALL;
    options.interactsSolverGroup = CollisionGroup.ALL;
    options.belongsCollisionGroup = CollisionGroup.TANK_HULL_PARTS;
    options.interactsCollisionGroup = CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_HULL_PARTS;

    caterpillars.forEach(([, set]) => {
        createThingFromParts({
            parentEId: tankEid,
            params: set,
            tag: TankPartCaterpillar,
        }, options);
    });

    Tank.setCaterpillarsLength(tankEid, caterpillars[0][0]);
}

export function createTankTurretParts(
    turretEid: number,
    { headSet, gunSet, bullet }: {
        headSet: PartsData[], gunSet: PartsData[], bullet: {
            position: [number, number],
            caliber: BulletCaliber,
        }
    },
    options: Options,
) {
    options.z = ZIndex.TankTurret;
    options.belongsCollisionGroup = CollisionGroup.TANK_TURRET_HEAD_PARTS;
    options.interactsCollisionGroup = CollisionGroup.ALL | CollisionGroup.WALL | CollisionGroup.TANK_TURRET_HEAD_PARTS | CollisionGroup.TANK_TURRET_GUN_PARTS;

    createThingFromParts({
        parentEId: turretEid,
        params: headSet,
    }, options);

    options.shadow[1] = 4;
    options.belongsCollisionGroup = CollisionGroup.TANK_TURRET_GUN_PARTS;
    options.interactsCollisionGroup = CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_TURRET_HEAD_PARTS | CollisionGroup.TANK_TURRET_GUN_PARTS;

    createThingFromParts({
        parentEId: turretEid,
        params: gunSet,
    }, options);

    TankTurret.setBulletData(turretEid, bullet.position, bullet.caliber);
}

const parentVector = new Vector2(0, 0);
const childVector = new Vector2(0, 0);

function createThingFromParts(
    {
        parentEId,
        params,
        tag,
    }: {
        parentEId: number,
        params: PartsData[],
        tag?: object
    },
    options: Options,
    { world, physicalWorld } = GameDI,
) {
    childVector.x = 0;
    childVector.y = 0;

    const rbId = RigidBodyRef.id[parentEId];
    const parentRb = physicalWorld.getRigidBody(rbId);
    const baseColor = options.color;

    for (let i = 0; i < params.length; i++) {
        const param = params[i];
        const parentTranslation = parentRb.translation();
        options.x = parentTranslation.x + param[0];
        options.y = parentTranslation.y + param[1];
        options.width = param[2];
        options.height = param[3];
        options.color = new Float32Array(baseColor);
        adjustBrightness(options.color, i / params.length / 3 - 0.1, i / params.length / 3 + 0.1);

        const [eid, pid] = createRectangleRR(options);

        parentVector.x = param[0];
        parentVector.y = param[1];
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
    }

    options.color = baseColor;
}

function adjustBrightness(color: TColor, start: number, end: number) {
    const factor = -1 * randomRangeFloat(start, end);
    color[0] = clamp(color[0] + factor, 0, 1);
    color[1] = clamp(color[1] + factor, 0, 1);
    color[2] = clamp(color[2] + factor, 0, 1);
}