import { createCircleRR } from './RigidRender.ts';
import { ActiveEvents, RigidBodyType } from '@dimforge/rapier2d-simd/rapier';
import {
    addTransformComponents,
    getMatrixRotationZ,
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../src/ECS/Components/Transform.ts';
import { GameDI } from '../../DI/GameDI.ts';
import { addComponent } from 'bitecs';
import { applyRotationToVector } from '../../Physical/applyRotationToVector.ts';
import { mat4, vec2, vec3 } from 'gl-matrix';
import { isNumber } from 'lodash-es';
import { Hitable } from './Hitable.ts';
import { Player } from './Player.ts';
import { Tank } from './Tank.ts';
import { random } from '../../../../../lib/random.ts';
import { ZIndex } from '../../consts.ts';
import { DestroyByTimeout } from './Destroy.ts';
import { CollisionGroup } from '../../Physical/createRigid.ts';

export const Bullet = {};

type Options = Parameters<typeof createCircleRR>[0];

const optionsBulletRR: Options = {
    x: 0,
    y: 0,
    z: ZIndex.Bullet,
    radius: 6,
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
    interactsCollisionGroup: CollisionGroup.ALL & ~CollisionGroup.TANK_GUN_PARTS,
};
const defaultOptionsBulletRR = structuredClone(optionsBulletRR);

const tmpSpeed = vec2.create();

export function createBulletRR(options: Partial<Options> & { speed: number, playerId: number }, { world } = GameDI) {
    Object.assign(optionsBulletRR, defaultOptionsBulletRR);
    Object.assign(optionsBulletRR, options);

    if (isNumber(options.speed) && options.speed > 0) {
        tmpSpeed[0] = 0;
        tmpSpeed[1] = -(options.speed ?? 0);
        applyRotationToVector(tmpSpeed, tmpSpeed, options.rotation ?? 0);
        optionsBulletRR.speedX = tmpSpeed[0];
        optionsBulletRR.speedY = tmpSpeed[1];
    }

    const [bulletId] = createCircleRR(optionsBulletRR);
    addComponent(world, bulletId, Bullet);
    addTransformComponents(world, bulletId);
    Player.addComponent(world, bulletId, options.playerId);
    Hitable.addComponent(world, bulletId);
    DestroyByTimeout.addComponent(world, bulletId, 8_000);

    return bulletId;
}


const optionsSpawnBullet = {
    x: 0,
    y: 0,
    color: new Float32Array(4).fill(1),
    rotation: 0,
    speed: 0,
    playerId: 0,
};
const tmpMatrix = mat4.create();
const tmpPosition = vec3.create() as Float32Array;

export function spawnBullet(tankId: number) {
    const globalTransform = GlobalTransform.matrix.getBatch(Tank.turretEId[tankId]);
    const bulletDelta = Tank.bulletStartPosition.getBatch(tankId);

    tmpPosition.set(bulletDelta);
    mat4.identity(tmpMatrix);
    mat4.translate(tmpMatrix, tmpMatrix, tmpPosition);
    mat4.multiply(tmpMatrix, globalTransform, tmpMatrix);

    optionsSpawnBullet.color[0] = random();
    optionsSpawnBullet.color[1] = random();
    optionsSpawnBullet.color[2] = random();
    optionsSpawnBullet.x = getMatrixTranslationX(tmpMatrix);
    optionsSpawnBullet.y = getMatrixTranslationY(tmpMatrix);
    optionsSpawnBullet.rotation = getMatrixRotationZ(tmpMatrix);
    optionsSpawnBullet.speed = Tank.bulletSpeed[tankId];
    optionsSpawnBullet.playerId = Player.id[tankId];

    createBulletRR(optionsSpawnBullet);
}