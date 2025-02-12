import { createCirceRR, createRectangleRR } from './RigidRender.ts';
import { RigidBodyType } from '@dimforge/rapier2d/src/dynamics/rigid_body.ts';
import { JointData, RigidBody, Vector2 } from '@dimforge/rapier2d';
import { addComponent, defineComponent, Types } from 'bitecs';
import { addTransformComponents } from '../../../../../src/ECS/Components/Transform.ts';
import { DI } from '../../DI';
import { Children } from './Children.ts';

export const Tank = defineComponent({
    bulletSpeed: Types.f64,
    bulletStartPosition: [Types.f64, 2],
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

type Options = Parameters<typeof createRectangleRR>[0] & Parameters<typeof createCirceRR>[0];

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
    belongsCollisionGroup: undefined,
    interactsCollisionGroup: undefined,
};

export const defaultOptions = { ...mutatedOptions };

const parentVector = new Vector2(0, 0);
const childVector = new Vector2(0, 0);
const createRectanglesRR = (
    parent: RigidBody,
    params: [number, number, number, number][],
    mutated: Options,
    x: number,
    y: number,
    physicalWorld = DI.physicalWorld,
) => {
    return params.map((param) => {
        const parentTranslation = parent.translation();
        mutated.x = parentTranslation.x + x + param[0];
        mutated.y = parentTranslation.y + y + param[1];
        mutated.width = param[2];
        mutated.height = param[3];
        const [wid, pid] = createRectangleRR(mutated);

        parentVector.x = x + param[0];
        parentVector.y = y + param[1];
        physicalWorld.createImpulseJoint(
            JointData.fixed(parentVector, 1, childVector, 1),
            parent,
            physicalWorld.getRigidBody(pid),
            true,
        );

        return wid;
    });
};

export function createTankRR(options: {
    x: number,
    y: number,
    rotation: number,
    color: [number, number, number, number],
}, { world, physicalWorld } = DI) {
    Object.assign(mutatedOptions, defaultOptions, options);

    mutatedOptions.radius = PADDING * 22;
    mutatedOptions.bodyType = RigidBodyType.Dynamic;
    mutatedOptions.belongsCollisionGroup = 0;
    mutatedOptions.interactsCollisionGroup = 0;
    mutatedOptions.color = [0.2, 0.1, 0.1, 0];

    const [tankId, rbId] = createCirceRR(mutatedOptions);
    const tankBody = physicalWorld.getRigidBody(rbId);

    const entitiesIds = new Float64Array(COMMON_LENGTH);

    mutatedOptions.rotation = 0;
    mutatedOptions.bodyType = RigidBodyType.Dynamic;
    mutatedOptions.belongsCollisionGroup = undefined;
    mutatedOptions.interactsCollisionGroup = undefined;

    // === TEST ===
    // entitiesIds.set(
    //     createRectanglesRR(tankBody, mainHullBase.slice(0, 1), mutatedOptions, 0, 0),
    //     0,
    // );

    // === Hull ===
    mutatedOptions.color = options.color;
    entitiesIds.set(
        createRectanglesRR(tankBody, mainHullBase, mutatedOptions, 0 - 3 * PADDING, 0 - 3 * PADDING),
        0,
    );

    // === Turret and Gun (8 прямоугольников) ===
    mutatedOptions.color = [0.5, 1, 0.5, 1];
    entitiesIds.set(
        createRectanglesRR(tankBody, turretAndGun, mutatedOptions, 0, 0 - 10 * PADDING),
        mainHullBase.length,
    );

    // === Left Track (13 прямоугольников) ===
    mutatedOptions.color = [0.5, 0.5, 0.5, 1];
    entitiesIds.set(
        createRectanglesRR(tankBody, caterpillar, mutatedOptions, 0 - 5 * PADDING, 0 - 4 * PADDING),
        mainHullBase.length + turretAndGun.length,
    );

    // === Right Track (13 прямоугольников) ===
    entitiesIds.set(
        createRectanglesRR(tankBody, caterpillar, mutatedOptions, 0 + 5 * PADDING, 0 - 4 * PADDING),
        mainHullBase.length + turretAndGun.length + caterpillar.length,
    );

    addTransformComponents(world, tankId);

    addComponent(world, Tank, tankId);
    Tank.bulletSpeed[tankId] = 300;
    Tank.bulletStartPosition[tankId][0] = PADDING / 2;
    Tank.bulletStartPosition[tankId][1] = -PADDING * 12;

    addComponent(world, Children, tankId);
    Children.entitiesCount[tankId] = entitiesIds.length;
    Children.entitiesIds[tankId] = entitiesIds;

    return tankId;
}
