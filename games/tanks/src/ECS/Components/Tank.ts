import { createCirceRR, createRectangleRR } from './RigidRender.ts';
import { RigidBodyType } from '@dimforge/rapier2d/src/dynamics/rigid_body.ts';
import { JointData, Vector2 } from '@dimforge/rapier2d';
import { addComponent, defineComponent, Types } from 'bitecs';
import { addTransformComponents } from '../../../../../src/ECS/Components/Transform.ts';
import { DI } from '../../DI';
import { Children } from './Children.ts';
import { CollisionGroup } from '../../Physical/createRigid.ts';
import { addPlayerComponent, getNewPlayerId } from './Player.ts';
import { addHitableComponent } from './Hitable.ts';
import { Parent } from './Parent.ts';
import { RigidBodyRef } from './Physical.ts';

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
    color: [1, 0, 0, 1],
    mass: 10,
    linearDamping: 5,
    angularDamping: 5,
    belongsCollisionGroup: CollisionGroup.TANK,
    interactsCollisionGroup: CollisionGroup.ALL,

    playerId: 0,
};

export const defaultOptions = { ...mutatedOptions };

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
    color: [number, number, number, number],
}, { world } = DI) {
    Object.assign(mutatedOptions, defaultOptions, options);

    mutatedOptions.radius = PADDING * 22;
    mutatedOptions.bodyType = RigidBodyType.Dynamic;
    mutatedOptions.belongsCollisionGroup = 0;
    mutatedOptions.interactsCollisionGroup = 0;
    mutatedOptions.color = [0.2, 0.1, 0.1, 0];
    mutatedOptions.playerId = getNewPlayerId();

    const [tankId] = createCirceRR(mutatedOptions);


    const partsEntityIds = new Float64Array(COMMON_LENGTH);

    mutatedOptions.rotation = 0;
    mutatedOptions.bodyType = RigidBodyType.Dynamic;
    mutatedOptions.belongsCollisionGroup = undefined;
    mutatedOptions.interactsCollisionGroup = undefined;

    // === Hull ===
    mutatedOptions.color = options.color;
    partsEntityIds.set(
        createRectanglesRR(tankId, mainHullBase, mutatedOptions, 0 - 3 * PADDING, 0 - 3 * PADDING),
        0,
    );

    // === Turret and Gun (8 прямоугольников) ===
    mutatedOptions.color = [0.5, 1, 0.5, 1];
    mutatedOptions.interactsCollisionGroup = CollisionGroup.WALL | CollisionGroup.TANK;
    partsEntityIds.set(
        createRectanglesRR(tankId, turretAndGun, mutatedOptions, 0, 0 - 10 * PADDING),
        mainHullBase.length,
    );
    mutatedOptions.interactsCollisionGroup = CollisionGroup.ALL;

    // === Left Track (13 прямоугольников) ===
    mutatedOptions.color = [0.5, 0.5, 0.5, 1];
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
