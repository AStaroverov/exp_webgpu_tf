import { createCircleRR, createRectangleRR } from './RigidRender.ts';
import { RigidBodyType } from '@dimforge/rapier2d/src/dynamics/rigid_body.ts';
import { JointData, Vector2 } from '@dimforge/rapier2d';
import { addComponent, defineComponent, removeComponent, Types } from 'bitecs';
import { addTransformComponents } from '../../../../../src/ECS/Components/Transform.ts';
import { DI } from '../../DI';
import { addChildrenComponent } from './Children.ts';
import { CollisionGroup } from '../../Physical/createRigid.ts';
import { addPlayerComponent, getNewPlayerId } from './Player.ts';
import { addHitableComponent } from './Hitable.ts';
import { Parent } from './Parent.ts';
import { RigidBodyRef } from './Physical.ts';
import { TColor } from '../../../../../src/ECS/Components/Common.ts';
import { createRectangleRigidGroup } from './RigidGroup.ts';
import { addTankControllerComponent } from './TankController.ts';
import { typicalRemoveEntity } from '../Utils/typicalRemoveEntity.ts';
import { addTankInputTensorComponent } from './TankState.ts';

export const Tank = defineComponent({
    turretEId: Types.f64,
    bulletSpeed: Types.f64,
    bulletStartPosition: [Types.f64, 2],
});

export const TankPart = defineComponent({
    jointPid: Types.f64,
});

const SIZE = 5;
const PADDING = SIZE + 1;
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

const caterpillarSet: [number, number, number, number][] =
    Array.from({ length: 26 }, (_, i) => {
        return [
            i * PADDING % (PADDING * 2), Math.floor(i / 2) * PADDING,
            SIZE, SIZE,
        ];
    });

const COMMON_LENGTH = hullSet.length + turretSet.length + gunSet.length + caterpillarSet.length * 2;

type Options = Parameters<typeof createRectangleRR>[0] & Parameters<typeof createCircleRR>[0] & {
    playerId: number,
};

export const mutatedOptions: Options = {
    x: 0,
    y: 0,
    width: 0,
    height: 0,
    radius: 0,
    rotation: 0,
    color: new Float32Array([1, 0, 0, 1]),
    shadow: new Float32Array([0, 3]),
    bodyType: RigidBodyType.Dynamic,
    density: 10,
    linearDamping: 5,
    angularDamping: 5,
    belongsSolverGroup: 0,
    interactsSolverGroup: 0,
    belongsCollisionGroup: 0,
    interactsCollisionGroup: 0,

    playerId: 0,
};

const defaultOptions = structuredClone(mutatedOptions);
const resetOptions = (target: Options, source: Parameters<typeof createTankRR>[0]) => {
    target.x = source?.x ?? defaultOptions.x;
    target.y = source?.y ?? defaultOptions.y;
    target.width = defaultOptions.width;
    target.height = defaultOptions.height;
    target.radius = defaultOptions.radius;
    target.rotation = source?.rotation ?? defaultOptions.rotation;
    (target.color as Float32Array).set(source?.color ?? defaultOptions.color, 0);
    (target.shadow as Float32Array).set(defaultOptions.shadow, 0);
    target.density = defaultOptions.density;
    target.linearDamping = defaultOptions.linearDamping;
    target.angularDamping = defaultOptions.angularDamping;
    target.belongsSolverGroup = defaultOptions.belongsSolverGroup;
    target.interactsSolverGroup = defaultOptions.interactsSolverGroup;
    target.belongsCollisionGroup = defaultOptions.belongsCollisionGroup;
    target.interactsCollisionGroup = defaultOptions.interactsCollisionGroup;
};
const updateColorOptions = (target: Options, color: TColor) => {
    (target.color as Float32Array).set(color, 0);
};

const parentVector = new Vector2(0, 0);
const childVector = new Vector2(0, 0);
const createRectanglesRR = (
    parentEId: number,
    params: [number, number, number, number][],
    options: Options,
    x: number,
    y: number,
    { world, physicalWorld } = DI,
) => {
    const rbId = RigidBodyRef.id[parentEId];
    const parentRb = physicalWorld.getRigidBody(rbId);

    childVector.x = 0;
    childVector.y = 0;

    return params.map((param) => {
        const parentTranslation = parentRb.translation();
        options.x = parentTranslation.x + x + param[0];
        options.y = parentTranslation.y + y + param[1];
        options.width = param[2];
        options.height = param[3];

        const [eid, pid] = createRectangleRR(options);

        parentVector.x = x + param[0];
        parentVector.y = y + param[1];
        const joint = physicalWorld.createImpulseJoint(
            JointData.fixed(parentVector, 0, childVector, 0),
            parentRb,
            physicalWorld.getRigidBody(pid),
            true,
        );

        addPlayerComponent(eid, options.playerId);
        addHitableComponent(world, eid);
        addComponent(world, TankPart, eid);
        TankPart.jointPid[eid] = joint.handle;
        addComponent(world, Parent, eid);
        Parent.id[eid] = parentEId;

        return eid;
    });
};


const DENSITY = 300;

export function createTankRR(options: {
    x: number,
    y: number,
    rotation: number,
    color: TColor,
}, { world, physicalWorld } = DI) {
    resetOptions(mutatedOptions, options);
    mutatedOptions.playerId = getNewPlayerId();

    mutatedOptions.density = DENSITY * 10;
    mutatedOptions.width = PADDING * 12;
    mutatedOptions.height = PADDING * 14;
    mutatedOptions.belongsCollisionGroup = CollisionGroup.TANK_BASE;
    mutatedOptions.interactsCollisionGroup = CollisionGroup.TANK_BASE;
    updateColorOptions(mutatedOptions, [0.5, 0.5, 0.5, 0.5]);
    // const [tankEid, tankPid] = createRectangleRR(mutatedOptions);
    const [tankEid, tankPid] = createRectangleRigidGroup(mutatedOptions);
    // {
    mutatedOptions.density = DENSITY;
    mutatedOptions.width = PADDING * 6;
    mutatedOptions.height = PADDING * 17;
    mutatedOptions.belongsCollisionGroup = 0;
    mutatedOptions.interactsCollisionGroup = 0;
    updateColorOptions(mutatedOptions, [0.5, 0, 0, 1]);
    // const [turretEid, turretPid] = createRectangleRR(mutatedOptions);
    const [turretEid, turretPid] = createRectangleRigidGroup(mutatedOptions);

    parentVector.x = 0;
    parentVector.y = 0;
    childVector.x = 0;
    childVector.y = PADDING * 5;
    const joint = physicalWorld.createImpulseJoint(
        JointData.revolute(parentVector, childVector),
        physicalWorld.getRigidBody(tankPid),
        physicalWorld.getRigidBody(turretPid),
        true,
    );

    addTransformComponents(world, turretEid);
    addComponent(world, TankPart, turretEid);
    TankPart.jointPid[turretEid] = joint.handle;
    addComponent(world, Parent, turretEid);
    Parent.id[turretEid] = tankEid;
    // }

    const partsEntityIds = new Float64Array(COMMON_LENGTH);

    mutatedOptions.density = DENSITY * 10;
    mutatedOptions.belongsSolverGroup = CollisionGroup.ALL;
    mutatedOptions.interactsSolverGroup = CollisionGroup.ALL;
    mutatedOptions.belongsCollisionGroup = CollisionGroup.TANK_BODY_PARTS;
    mutatedOptions.interactsCollisionGroup = CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_BODY_PARTS;

    // === Hull ===
    updateColorOptions(mutatedOptions, options.color);
    partsEntityIds.set(
        createRectanglesRR(tankEid, hullSet, mutatedOptions, 0 - 3.5 * PADDING, 0 - 5 * PADDING),
        0,
    );

    // === Left Track (13 прямоугольников) ===
    updateColorOptions(mutatedOptions, [0.5, 0.5, 0.5, 1]);
    partsEntityIds.set(
        createRectanglesRR(tankEid, caterpillarSet, mutatedOptions, 0 - 5.5 * PADDING, 0 - 6 * PADDING),
        hullSet.length,
    );

    // === Right Track (13 прямоугольников) ===
    updateColorOptions(mutatedOptions, [0.5, 0.5, 0.5, 1]);
    partsEntityIds.set(
        createRectanglesRR(tankEid, caterpillarSet, mutatedOptions, 0 + 4.5 * PADDING, 0 - 6 * PADDING),
        hullSet.length + caterpillarSet.length,
    );

    // // === Turret and Gun (8 прямоугольников) ===
    mutatedOptions.density = DENSITY;
    mutatedOptions.belongsCollisionGroup = CollisionGroup.TANK_TURRET_PARTS;
    mutatedOptions.interactsCollisionGroup = CollisionGroup.ALL | CollisionGroup.WALL | CollisionGroup.TANK_TURRET_PARTS | CollisionGroup.TANK_GUN_PARTS;
    updateColorOptions(mutatedOptions, [0.5, 1, 0.5, 1]);
    partsEntityIds.set(
        createRectanglesRR(turretEid, turretSet, mutatedOptions, 0 - 0.5 * PADDING, 0 - 8 * PADDING),
        hullSet.length + caterpillarSet.length * 2,
    );
    mutatedOptions.shadow[1] = 4;
    mutatedOptions.belongsCollisionGroup = CollisionGroup.TANK_GUN_PARTS;
    mutatedOptions.interactsCollisionGroup = CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_TURRET_PARTS | CollisionGroup.TANK_GUN_PARTS;
    partsEntityIds.set(
        createRectanglesRR(turretEid, gunSet, mutatedOptions, 0 - 0.5 * PADDING, 0 - 8 * PADDING),
        hullSet.length + turretSet.length + caterpillarSet.length * 2,
    );


    addComponent(world, Tank, tankEid);
    Tank.turretEId[tankEid] = turretEid;
    Tank.bulletSpeed[tankEid] = 300;
    Tank.bulletStartPosition[tankEid][0] = PADDING / 2;
    Tank.bulletStartPosition[tankEid][1] = -PADDING * 11;

    addTransformComponents(world, tankEid);
    addTankControllerComponent(world, tankEid);
    addChildrenComponent(tankEid, COMMON_LENGTH, partsEntityIds);
    addPlayerComponent(tankEid, mutatedOptions.playerId);

    // for ML learning
    addTankInputTensorComponent(tankEid);

    return tankEid;
}

export function removeTankComponentsWithoutParts(tankEid: number) {
    const turretEid = Tank.turretEId[tankEid];
    typicalRemoveEntity(tankEid);
    typicalRemoveEntity(turretEid);
}

export function removeTankPartJointComponent(tankPartEid: number, { world } = DI) {
    TankPart.jointPid[tankPartEid] = -1;
    removeComponent(world, TankPart, tankPartEid);
}