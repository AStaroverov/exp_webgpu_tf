import { createRectangleRR } from './RigidRender.ts';
import { RigidBodyType } from '@dimforge/rapier2d/src/dynamics/rigid_body.ts';
import {
    addTransformComponents,
    getMatrixRotationZ,
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../src/ECS/Components/Transform.ts';
import { DI } from '../../DI';
import { addComponent, defineComponent } from 'bitecs';
import { applyRotationToVector } from '../../Physical/applyRotationToVector.ts';
import { mat4, vec2, vec3 } from 'gl-matrix';
import { isNumber } from 'lodash-es';
import { ActiveEvents } from '@dimforge/rapier2d';
import { CollisionGroup } from '../../Physical/createRigid.ts';
import { addHitableComponent } from './Hitable.ts';
import { addPlayerComponent, Player } from './Player.ts';
import { Tank } from './Tank.ts';

export const Bullet = defineComponent();

type Options = Parameters<typeof createRectangleRR>[0] & { speed: number, playerId: number };

export const mutatedOptions: Options = {
    x: 0,
    y: 0,
    width: 5,
    height: 5,
    speedX: 0,
    speedY: 0,
    rotation: 0,
    color: new Float32Array([1, 0, 0, 1]),
    shadow: new Float32Array([0, 2]),
    bodyType: RigidBodyType.Dynamic,
    density: 10,
    angularDamping: 0.1,
    linearDamping: 0.1,
    collisionEvent: ActiveEvents.CONTACT_FORCE_EVENTS,
    belongsCollisionGroup: CollisionGroup.BULLET,
    interactsCollisionGroup: CollisionGroup.ALL & ~CollisionGroup.TANK_GUN,

    //
    speed: 0,
    playerId: 0,
};

const tmpSpeed = vec2.create();

export function createBulletRR(options: Options, { world } = DI) {
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


const tmpMatrix = mat4.create();
const tmpPosition = vec3.create() as Float32Array;

export function spawnBullet(tankId: number) {
    const globalTransform = GlobalTransform.matrix[Tank.turretEId[tankId]];
    const bulletDelta = Tank.bulletStartPosition[tankId];

    tmpPosition.set(bulletDelta);
    mat4.identity(tmpMatrix);
    mat4.translate(tmpMatrix, tmpMatrix, tmpPosition);
    mat4.multiply(tmpMatrix, globalTransform, tmpMatrix);

    mutatedOptions.x = getMatrixTranslationX(tmpMatrix);
    mutatedOptions.y = getMatrixTranslationY(tmpMatrix);
    mutatedOptions.rotation = getMatrixRotationZ(tmpMatrix);
    mutatedOptions.speed = Tank.bulletSpeed[tankId];
    mutatedOptions.playerId = Player.id[tankId];

    createBulletRR(mutatedOptions);
}