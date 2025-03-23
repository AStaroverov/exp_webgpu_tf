import { createRectangleRR } from './RigidRender.ts';
import { RigidBodyType } from '@dimforge/rapier2d/src/dynamics/rigid_body.ts';
import { JointData, Vector2 } from '@dimforge/rapier2d';
import { addComponent } from 'bitecs';
import { addTransformComponents } from '../../../../../src/ECS/Components/Transform.ts';
import { GameDI } from '../../DI/GameDI.ts';
import { addChildren, addChildrenComponent, Children, removeChild } from './Children.ts';
import { CollisionGroup } from '../../Physical/createRigid.ts';
import { addPlayerComponent, getNewPlayerId } from './Player.ts';
import { Hitable } from './Hitable.ts';
import { addParentComponent, Parent } from './Parent.ts';
import { RigidBodyRef } from './Physical.ts';
import { TColor } from '../../../../../src/ECS/Components/Common.ts';
import { createRectangleRigidGroup } from './RigidGroup.ts';
import { TankController } from './TankController.ts';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.ts';
import { addTankInputTensorComponent } from './TankState.ts';
import { delegate } from '../../../../../src/delegate.ts';
import { NestedArray, TypedArray } from '../../../../../src/utils.ts';
import { ZIndex } from '../../consts.ts';
import { min, smoothstep } from '../../../../../lib/math.ts';
import { component } from '../../../../../src/ECS/utils.ts';
import { createCircle } from '../../../../../src/ECS/Entities/Shapes.ts';

export const TANK_APPROXIMATE_COLLISION_RADIUS = 80;

export const Tank = component({
    aimEid: TypedArray.f64(delegate.defaultSize),
    turretEId: TypedArray.f64(delegate.defaultSize),
    bulletSpeed: TypedArray.f64(delegate.defaultSize),
    bulletStartPosition: NestedArray.f64(2, delegate.defaultSize),
    initialPartsCount: TypedArray.f64(delegate.defaultSize),
});

export const TankPart = ({
    jointPid: TypedArray.f64(delegate.defaultSize),
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

const PARTS_COUNT = hullSet.length + turretSet.length + gunSet.length + caterpillarSet.length * 2;

const mutatedOptions = {
    x: 0,
    y: 0,
    z: 0,
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
type Options = typeof mutatedOptions;

const defaultOptions = structuredClone(mutatedOptions);
const resetOptions = (target: Options, source: Parameters<typeof createTankRR>[0]) => {
    target.x = source?.x ?? defaultOptions.x;
    target.y = source?.y ?? defaultOptions.y;
    target.z = defaultOptions.z;
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
    { world, physicalWorld } = GameDI,
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
        Hitable.addComponent(eid);
        addParentComponent(eid, parentEId);
        addChildren(parentEId, eid);

        addComponent(world, eid, TankPart);
        TankPart.jointPid[eid] = joint.handle;

        return eid;
    });
};


const DENSITY = 300;

export function createTankRR(options: {
    x: number,
    y: number,
    rotation: number,
    color: TColor,
}, { world, physicalWorld } = GameDI) {
    resetOptions(mutatedOptions, options);
    mutatedOptions.playerId = getNewPlayerId();

    mutatedOptions.density = DENSITY * 10;
    mutatedOptions.width = PADDING * 12;
    mutatedOptions.height = PADDING * 14;
    mutatedOptions.belongsCollisionGroup = CollisionGroup.TANK_BASE;
    mutatedOptions.interactsCollisionGroup = CollisionGroup.TANK_BASE;
    // updateColorOptions(mutatedOptions, [0.5, 0.5, 0.5, 0.5]);
    // const [tankEid, tankPid] = createRectangleRR(mutatedOptions);
    const [tankEid, tankPid] = createRectangleRigidGroup(mutatedOptions);
    addComponent(world, tankEid, Tank);
    Tank.bulletSpeed[tankEid] = 300;
    Tank.bulletStartPosition.set(tankEid, 0, 0);
    Tank.bulletStartPosition.set(tankEid, 1, -PADDING * 9);
    Tank.initialPartsCount[tankEid] = PARTS_COUNT;
    addChildrenComponent(tankEid);
    addTransformComponents(world, tankEid);
    TankController.addComponent(tankEid);
    addPlayerComponent(tankEid, mutatedOptions.playerId);
    addTankInputTensorComponent(tankEid);

    mutatedOptions.radius = 16;
    const aimEid = createCircle(GameDI.world, mutatedOptions);
    addParentComponent(aimEid, tankEid);
    addChildren(tankEid, aimEid);

    Tank.aimEid[tankEid] = aimEid;


    // {
    mutatedOptions.density = DENSITY;
    mutatedOptions.width = PADDING * 6;
    mutatedOptions.height = PADDING * 17;
    mutatedOptions.belongsCollisionGroup = 0;
    mutatedOptions.interactsCollisionGroup = 0;
    updateColorOptions(mutatedOptions, [0.5, 0, 0, 1]);
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
    addComponent(world, turretEid, TankPart);
    TankPart.jointPid[turretEid] = joint.handle;
    addComponent(world, turretEid, Parent);
    Parent.id[turretEid] = tankEid;
    addChildrenComponent(turretEid);
    // }

    Tank.turretEId[tankEid] = turretEid;
    addChildren(tankEid, turretEid);

    mutatedOptions.z = ZIndex.TankHull;
    mutatedOptions.density = DENSITY * 10;
    mutatedOptions.belongsSolverGroup = CollisionGroup.ALL;
    mutatedOptions.interactsSolverGroup = CollisionGroup.ALL;
    mutatedOptions.belongsCollisionGroup = CollisionGroup.TANK_HULL_PARTS;
    mutatedOptions.interactsCollisionGroup = CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_HULL_PARTS;

    // === Hull ===
    updateColorOptions(mutatedOptions, options.color);
    createRectanglesRR(tankEid, hullSet, mutatedOptions, 0 - 3.5 * PADDING, 0 - 5 * PADDING);

    // === Left Track (13 прямоугольников) ===
    updateColorOptions(mutatedOptions, [0.5, 0.5, 0.5, 1]);
    createRectanglesRR(tankEid, caterpillarSet, mutatedOptions, 0 - 5.5 * PADDING, 0 - 6 * PADDING);

    // === Right Track (13 прямоугольников) ===
    updateColorOptions(mutatedOptions, [0.5, 0.5, 0.5, 1]);
    createRectanglesRR(tankEid, caterpillarSet, mutatedOptions, 0 + 4.5 * PADDING, 0 - 6 * PADDING);

    // === Turret and Gun (8 прямоугольников) ===
    mutatedOptions.z = ZIndex.TankTurret;
    mutatedOptions.density = DENSITY;
    mutatedOptions.belongsCollisionGroup = CollisionGroup.TANK_TURRET_PARTS;
    mutatedOptions.interactsCollisionGroup = CollisionGroup.ALL | CollisionGroup.WALL | CollisionGroup.TANK_TURRET_PARTS | CollisionGroup.TANK_GUN_PARTS;
    updateColorOptions(mutatedOptions, [0.5, 1, 0.5, 1]);
    createRectanglesRR(turretEid, turretSet, mutatedOptions, 0 - 0.5 * PADDING, 0 - 8 * PADDING);

    mutatedOptions.shadow[1] = 4;
    mutatedOptions.belongsCollisionGroup = CollisionGroup.TANK_GUN_PARTS;
    mutatedOptions.interactsCollisionGroup = CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_TURRET_PARTS | CollisionGroup.TANK_GUN_PARTS;
    createRectanglesRR(turretEid, gunSet, mutatedOptions, 0 - 0.5 * PADDING, 0 - 8 * PADDING);

    return tankEid;
}

export function removeTankComponentsWithoutParts(tankEid: number) {
    const aimEid = Tank.aimEid[tankEid];
    const turretEid = Tank.turretEId[tankEid];
    removeChild(tankEid, aimEid);
    removeChild(tankEid, turretEid);
    scheduleRemoveEntity(aimEid, false);
    scheduleRemoveEntity(turretEid, false);
    scheduleRemoveEntity(tankEid, false);
}

export function resetTankPartJointComponent(tankPartEid: number) {
    TankPart.jointPid[tankPartEid] = -1;
}

export function getTankCurrentPartsCount(tankEid: number) {
    return Children.entitiesCount[tankEid] + Children.entitiesCount[Tank.turretEId[tankEid]];
}

export const HEALTH_THRESHOLD = 0.75;

// return from 0 to 1
export function getTankHealth(tankEid: number): number {
    const initialPartsCount = Tank.initialPartsCount[tankEid];
    const partsCount = getTankCurrentPartsCount(tankEid);
    const absHealth = min(1, partsCount / initialPartsCount);
    const health = smoothstep(HEALTH_THRESHOLD, 1, absHealth);

    return health;
}