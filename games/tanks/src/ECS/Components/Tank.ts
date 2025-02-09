import { createRectangleRR } from './RigidRender.ts';
import { RigidBodyType } from '@dimforge/rapier2d/src/dynamics/rigid_body.ts';
import { ActiveEvents } from '@dimforge/rapier2d';
import { addComponent, addEntity } from 'bitecs';
import {
    addTransformComponents,
    LocalTransform,
    setMatrixRotateZ,
    setMatrixTranslate,
} from '../../../../../src/ECS/Components/Transform.ts';
import { DI } from '../../DI';
import { Children } from './Children.ts';

type Options = Parameters<typeof createRectangleRR>[0] & { scale: number };

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

const createRectanglesRR = (
    params: [number, number, number, number][],
    mutated: Options,
    x: number,
    y: number,
) => {
    return params.map((param) => {
        mutated.x = x + param[0];
        mutated.y = y + param[1];
        mutated.width = param[2];
        mutated.height = param[3];
        return createRectangleRR(mutated);
    });
};

export const mutatedOptions: Options = {
    x: 0,
    y: 0,
    scale: 1,
    rotation: 0,
    color: [1, 1, 1, 1],
    width: 0,
    height: 0,
    bodyType: RigidBodyType.KinematicPositionBased,
    gravityScale: 1,
    mass: 10,
    collisionEvent: ActiveEvents.COLLISION_EVENTS,
};

export function createTankRR(options: {
    x: number,
    y: number,
    scale: number,
    rotation: number,
    color: [number, number, number, number],
}, { world } = DI) {
    const entitiesIds = new Float64Array(COMMON_LENGTH);
    const rotation = options.rotation;
    Object.assign(mutatedOptions, options, {
        rotation: 0,
        width: 0,
        height: 0,
        bodyType: RigidBodyType.KinematicPositionBased,
        gravityScale: 1,
        mass: 10,
        collisionEvent: ActiveEvents.COLLISION_EVENTS,
    });

    // === Hull ===
    entitiesIds.set(
        createRectanglesRR(mainHullBase, mutatedOptions, 0 - 3 * PADDING, 0 - 3 * PADDING),
        0,
    );

    // === Turret and Gun (8 прямоугольников) ===
    mutatedOptions.color = [0.5, 1, 0.5, 1];
    entitiesIds.set(
        createRectanglesRR(turretAndGun, mutatedOptions, 0, 0 - 11 * PADDING),
        mainHullBase.length,
    );

    // === Left Track (13 прямоугольников) ===
    mutatedOptions.color = [0.5, 0.5, 0.5, 1];
    entitiesIds.set(
        createRectanglesRR(caterpillar, mutatedOptions, 0 - 5 * PADDING, 0 - 4 * PADDING),
        mainHullBase.length + turretAndGun.length,
    );

    // === Right Track (13 прямоугольников) ===
    entitiesIds.set(
        createRectanglesRR(caterpillar, mutatedOptions, 0 + 5 * PADDING, 0 - 4 * PADDING),
        mainHullBase.length + turretAndGun.length + caterpillar.length,
    );

    const tankId = addEntity(world);
    addComponent(world, Children, tankId);
    Children.entitiesIds[tankId] = entitiesIds;
    addTransformComponents(world, tankId);
    setMatrixTranslate(LocalTransform.matrix[tankId], options.x, options.y);
    setMatrixRotateZ(LocalTransform.matrix[tankId], rotation);

    return tankId;
}
