import { createCirceRR, createRectangleRR } from './RigidRender.ts';
import { RigidBodyType } from '@dimforge/rapier2d/src/dynamics/rigid_body.ts';
import { JointData, Vector2 } from '@dimforge/rapier2d';
import { addComponent, defineComponent, Types } from 'bitecs';
import { addTransformComponents } from '../../../../../src/ECS/Components/Transform.ts';
import { DI } from '../../DI';
import { Children } from './Children.ts';
import { CollisionGroup, createRigidCircle } from '../../Physical/createRigid.ts';
import { addPlayerComponent, getNewPlayerId } from './Player.ts';
import { addHitableComponent } from './Hitable.ts';
import { Parent } from './Parent.ts';
import { RigidBodyRef } from './Physical.ts';
import { TColor } from '../../../../../src/ECS/Components/Common.ts';
import { createRigidGroup } from './RigidGroup.ts';

export const Tank = defineComponent({
    bulletSpeed: Types.f64,
    bulletStartPosition: [Types.f64, 2],
});

export const TankPart = defineComponent({
    jointId: Types.f64,
});

const SIZE = 5;
const PADDING = SIZE + 1;
const mainHullBase: [number, number, number, number][] = Array.from({ length: 88 }, (_, i) => {
    return [
        i * PADDING % (PADDING * 8), Math.floor(i / 8) * PADDING,
        SIZE, SIZE,
    ];
});

const turretAndGun: [number, number, number, number][] = [
    ...Array.from({ length: 20 }, (_, i): [number, number, number, number] => {
        return [
            i * PADDING % (PADDING * 2), Math.floor(i / 2) * PADDING,
            SIZE, SIZE,
        ];
    }),
    ...Array.from({ length: 42 }, (_, i): [number, number, number, number] => {
        return [
            -PADDING * 2 + i * PADDING % (PADDING * 6), 10 * PADDING + Math.floor(i / 6) * PADDING,
            SIZE, SIZE,
        ];
    }),
];

const caterpillar: [number, number, number, number][] = Array.from({ length: 26 }, (_, i) => {
    return [
        i * PADDING % (PADDING * 2), Math.floor(i / 2) * PADDING,
        SIZE, SIZE,
    ];
});

const COMMON_LENGTH = mainHullBase.length + turretAndGun.length + caterpillar.length * 2;

type Options = Parameters<typeof createRectangleRR>[0] & Parameters<typeof createCirceRR>[0] & {
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
    mass: 10,
    linearDamping: 5,
    angularDamping: 5,
    belongsCollisionGroup: CollisionGroup.TANK,
    interactsCollisionGroup: CollisionGroup.ALL,

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
    target.mass = defaultOptions.mass;
    target.linearDamping = defaultOptions.linearDamping;
    target.angularDamping = defaultOptions.angularDamping;
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
    coords: [number, number, number, number][],
    options: Options,
    x: number,
    y: number,
    { world, physicalWorld } = DI,
) => {
    const rbId = RigidBodyRef.id[parentEId];
    const parentRb = physicalWorld.getRigidBody(rbId);

    return coords.map((param) => {
        const parentTranslation = parentRb.translation();
        options.x = parentTranslation.x + x + param[0];
        options.y = parentTranslation.y + y + param[1];
        options.width = param[2];
        options.height = param[3];

        const [eid, pid] = createRectangleRR(options);

        parentVector.x = x + param[0];
        parentVector.y = y + param[1];
        const joint = physicalWorld.createImpulseJoint(
            JointData.fixed(parentVector, 10, childVector, 10),
            parentRb,
            physicalWorld.getRigidBody(pid),
            true,
        );

        addPlayerComponent(world, eid, options.playerId);
        addHitableComponent(world, eid);
        addComponent(world, TankPart, eid);
        TankPart.jointId[eid] = joint.handle;
        addComponent(world, Parent, eid);
        Parent.id[eid] = parentEId;

        return eid;
    });
};

export function createTankRR(options: {
    x: number,
    y: number,
    rotation: number,
    color: TColor,
}, { world } = DI) {
    resetOptions(mutatedOptions, options);
    mutatedOptions.playerId = getNewPlayerId();
    mutatedOptions.radius = PADDING * 22;
    mutatedOptions.belongsCollisionGroup = 0;
    mutatedOptions.interactsCollisionGroup = 0;

    const [tankId] = createRigidGroup(createRigidCircle(mutatedOptions));
    const partsEntityIds = new Float64Array(COMMON_LENGTH);

    mutatedOptions.rotation = 0;
    mutatedOptions.belongsCollisionGroup = undefined;
    mutatedOptions.interactsCollisionGroup = undefined;

    // === Hull ===
    updateColorOptions(mutatedOptions, options.color);
    partsEntityIds.set(
        createRectanglesRR(tankId, mainHullBase, mutatedOptions, 0 - 3 * PADDING, 0 - 3 * PADDING),
        0,
    );

    // === Turret and Gun (8 прямоугольников) ===
    updateColorOptions(mutatedOptions, [0.5, 1, 0.5, 1]);
    mutatedOptions.interactsCollisionGroup = CollisionGroup.WALL | CollisionGroup.TANK;
    partsEntityIds.set(
        createRectanglesRR(tankId, turretAndGun, mutatedOptions, 0, 0 - 10 * PADDING),
        mainHullBase.length,
    );
    mutatedOptions.interactsCollisionGroup = CollisionGroup.ALL;

    // === Left Track (13 прямоугольников) ===
    updateColorOptions(mutatedOptions, [0.5, 0.5, 0.5, 1]);
    partsEntityIds.set(
        createRectanglesRR(tankId, caterpillar, mutatedOptions, 0 - 5 * PADDING, 0 - 4 * PADDING),
        mainHullBase.length + turretAndGun.length,
    );

    // === Right Track (13 прямоугольников) ===
    partsEntityIds.set(
        createRectanglesRR(tankId, caterpillar, mutatedOptions, 0 + 5 * PADDING, 0 - 4 * PADDING),
        mainHullBase.length + turretAndGun.length + caterpillar.length,
    );

    addTransformComponents(world, tankId);

    addComponent(world, Tank, tankId);
    Tank.bulletSpeed[tankId] = 300;
    Tank.bulletStartPosition[tankId][0] = PADDING / 2;
    Tank.bulletStartPosition[tankId][1] = -PADDING * 12;

    addComponent(world, Children, tankId);
    Children.entitiesCount[tankId] = partsEntityIds.length;
    Children.entitiesIds[tankId] = partsEntityIds;

    addPlayerComponent(world, tankId, mutatedOptions.playerId);

    return tankId;
}
