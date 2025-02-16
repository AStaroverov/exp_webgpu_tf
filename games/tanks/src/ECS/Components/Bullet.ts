import { createRectangleRR } from './RigidRender.ts';
import { RigidBodyType } from '@dimforge/rapier2d/src/dynamics/rigid_body.ts';
import { addTransformComponents } from '../../../../../src/ECS/Components/Transform.ts';
import { DI } from '../../DI';
import { addComponent, defineComponent } from 'bitecs';
import { applyRotationToVector } from '../../Physical/applyRotationToVector.ts';
import { vec2 } from 'gl-matrix';
import { isNumber } from 'lodash-es';
import { ActiveEvents } from '@dimforge/rapier2d';
import { CollisionGroup } from '../../Physical/createRigid.ts';
import { addHitableComponent } from './Hitable.ts';
import { addPlayerComponent } from './Player.ts';

export const Bullet = defineComponent();

type Options = Parameters<typeof createRectangleRR>[0]

export const mutatedOptions: Options = {
    x: 0,
    y: 0,
    width: 10,
    height: 10,
    speedX: 0,
    speedY: 0,
    rotation: 0,
    color: new Float32Array([1, 1, 1, 1]),
    shadow: new Float32Array([0, 2]),
    bodyType: RigidBodyType.Dynamic,
    mass: 10,
    angularDamping: 0.1,
    linearDamping: 0.1,
    collisionEvent: ActiveEvents.CONTACT_FORCE_EVENTS,
    belongsCollisionGroup: CollisionGroup.BULLET,
};

const tmpSpeed = vec2.create();

export function createBulletRR(options: Options & { speed: number, playerId: number }, { world } = DI) {
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
    addPlayerComponent(world, bulletId, options.playerId);
    addHitableComponent(world, bulletId);
    addTransformComponents(world, bulletId);

    return bulletId;
}
