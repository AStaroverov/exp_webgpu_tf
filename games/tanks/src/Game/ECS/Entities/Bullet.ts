import { GameDI } from '../../DI/GameDI.ts';
import { isNumber } from 'lodash-es';
import { applyRotationToVector } from '../../Physical/applyRotationToVector.ts';
import { createRectangleRR } from '../Components/RigidRender.ts';
import { PlayerRef } from '../Components/PlayerRef.ts';
import { Hitable } from '../Components/Hitable.ts';
import { DestroyByTimeout } from '../Components/Destroy.ts';
import { mat4, vec2, vec3 } from 'gl-matrix';
import {
    getMatrixRotationZ,
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../../src/ECS/Components/Transform.ts';
import { Tank } from '../Components/Tank.ts';
import { ZIndex } from '../../consts.ts';
import { ActiveEvents, RigidBodyType } from '@dimforge/rapier2d-simd';
import { CollisionGroup } from '../../Physical/createRigid.ts';
import { Bullet, BulletCaliber, mapBulletCaliber, MAX_BULLET_SPEED } from '../Components/Bullet.ts';
import { TeamRef } from '../Components/TeamRef.ts';
import { Color } from '../../../../../../src/ECS/Components/Common.ts';
import { TankTurret } from '../Components/TankTurret.ts';
import { min } from '../../../../../../lib/math.ts';

type Options = Parameters<typeof createRectangleRR>[0];
const optionsBulletRR: Options = {
    x: 0,
    y: 0,
    z: ZIndex.Bullet,
    width: 0,
    height: 0,
    speedX: 0,
    speedY: 0,
    rotation: 0,
    color: new Float32Array([1, 0, 0, 1]),
    shadow: new Float32Array([0, 2]),
    bodyType: RigidBodyType.Dynamic,
    density: 10_000,
    angularDamping: 0.1,
    linearDamping: 0.1,
    collisionEvent: ActiveEvents.CONTACT_FORCE_EVENTS,
    belongsCollisionGroup: CollisionGroup.BULLET,
    interactsCollisionGroup: CollisionGroup.ALL & ~CollisionGroup.TANK_TURRET_GUN_PARTS,
};
const defaultOptionsBulletRR = structuredClone(optionsBulletRR);
const tmpSpeed = vec2.create();

export function createBullet(options: Partial<Options> & {
    speed: number,
    playerId: number,
    teamId: number
}, { world } = GameDI) {
    Object.assign(optionsBulletRR, defaultOptionsBulletRR);
    Object.assign(optionsBulletRR, options);

    if (isNumber(options.speed) && options.speed > 0) {
        tmpSpeed[0] = 0;
        tmpSpeed[1] = -(options.speed ?? 0);
        applyRotationToVector(tmpSpeed, tmpSpeed, options.rotation ?? 0);
        optionsBulletRR.speedX = tmpSpeed[0];
        optionsBulletRR.speedY = tmpSpeed[1];
    }

    const [bulletId] = createRectangleRR(optionsBulletRR);
    Bullet.addComponent(world, bulletId);
    TeamRef.addComponent(world, bulletId, options.teamId);
    PlayerRef.addComponent(world, bulletId, options.playerId);
    Hitable.addComponent(world, bulletId);
    DestroyByTimeout.addComponent(world, bulletId, 8_000);

    return bulletId;
}

const optionsSpawnBullet = {
    x: 0,
    y: 0,
    width: 0,
    height: 0,
    color: new Float32Array(4).fill(1),
    speed: 0,
    density: 0,
    rotation: 0,
    playerId: 0,
    teamId: 0,
};
const tmpMatrix = mat4.create();
const tmpPosition = vec3.create() as Float32Array;

export function spawnBullet(tankEid: number) {
    const turretEid = Tank.turretEId[tankEid];
    const globalTransform = GlobalTransform.matrix.getBatch(turretEid);
    const bulletPosition = TankTurret.bulletStartPosition.getBatch(turretEid);
    const bulletCaliber = mapBulletCaliber[TankTurret.bulletCaliber[turretEid] as BulletCaliber];

    tmpPosition.set(bulletPosition);
    mat4.identity(tmpMatrix);
    mat4.translate(tmpMatrix, tmpMatrix, tmpPosition);
    mat4.multiply(tmpMatrix, globalTransform, tmpMatrix);

    Color.applyColorToArray(tankEid, optionsSpawnBullet.color);
    optionsSpawnBullet.x = getMatrixTranslationX(tmpMatrix);
    optionsSpawnBullet.y = getMatrixTranslationY(tmpMatrix);
    optionsSpawnBullet.width = bulletCaliber.width;
    optionsSpawnBullet.height = bulletCaliber.height;
    optionsSpawnBullet.rotation = getMatrixRotationZ(tmpMatrix);
    optionsSpawnBullet.speed = min(bulletCaliber.speed, MAX_BULLET_SPEED);
    optionsSpawnBullet.density = bulletCaliber.density;
    optionsSpawnBullet.teamId = TeamRef.id[tankEid];
    optionsSpawnBullet.playerId = PlayerRef.id[tankEid];

    createBullet(optionsSpawnBullet);
}