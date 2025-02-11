import { createRectangleRR } from './RigidRender.ts';
import { RigidBodyType } from '@dimforge/rapier2d/src/dynamics/rigid_body.ts';
import { addTransformComponents } from '../../../../../src/ECS/Components/Transform.ts';
import { DI } from '../../DI';
import { addComponent, defineComponent } from 'bitecs';
import { applyRotationToVector } from '../../Physical/applyRotationToVector.ts';
import { vec2 } from 'gl-matrix';
import { isNumber } from 'lodash-es';

export const Bullet = defineComponent();

type Options = Parameters<typeof createRectangleRR>[0]

export const mutatedOptions: Options = {
    x: 0,
    y: 0,
    speedX: 0,
    speedY: 0,
    rotation: 0,
    color: [1, 1, 1, 1],
    width: 10,
    height: 10,
    bodyType: RigidBodyType.Dynamic,
    gravityScale: 0,
    mass: 1,
};

const tmpSpeed = vec2.create();

export function createBulletRR(options: {
    x?: number,
    y?: number,
    speed?: number,
    rotation?: number,
    color: [number, number, number, number],
}, { world } = DI) {
    Object.assign(mutatedOptions, options);

    if (isNumber(options.speed) && options.speed > 0) {
        tmpSpeed[0] = 0;
        tmpSpeed[1] = -(options.speed ?? 0);
        applyRotationToVector(tmpSpeed, tmpSpeed, options.rotation ?? 0);
        mutatedOptions.speedX = tmpSpeed[0];
        mutatedOptions.speedY = tmpSpeed[1];
    }

    const [bulletId] = createRectangleRR(mutatedOptions);
    addComponent(world, Bullet, bulletId);
    addTransformComponents(world, bulletId);

    return bulletId;
}
