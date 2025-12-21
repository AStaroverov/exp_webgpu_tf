import { GameDI } from '../../DI/GameDI.ts';
import { isNumber } from 'lodash-es';
import { applyRotationToVector } from '../../Physical/applyRotationToVector.ts';
import { createRectangleRR } from '../Components/RigidRender.ts';
import { PlayerRef } from '../Components/PlayerRef.ts';
import { Hitable } from '../Components/Hitable.ts';
import { DestroyBySpeed } from '../Components/Destroy.ts';
import { spawnMuzzleFlash } from './MuzzleFlash.ts';
import { SoundType } from '../Components/Sound.ts';
import { spawnSoundAtPosition } from './Sound.ts';
import { mat4, vec2, vec3 } from 'gl-matrix';
import {
    getMatrixRotationZ,
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { Firearms } from '../Components/Firearms.ts';
import { Tank } from '../Components/Tank.ts';
import { ZIndex } from '../../consts.ts';
import { ActiveEvents, RigidBodyType } from '@dimforge/rapier2d-simd';
import { CollisionGroup } from '../../Physical/createRigid.ts';
import { Bullet, BulletCaliber, mapBulletCaliber, MAX_BULLET_SPEED, MIN_BULLET_SPEED } from '../Components/Bullet.ts';
import { TeamRef } from '../Components/TeamRef.ts';
import { Color } from '../../../../../renderer/src/ECS/Components/Common.ts';
import { min } from '../../../../../../lib/math.ts';
import { Damagable } from '../Components/Damagable.ts';
import { ExplosionConfig, SoundConfig } from '../../Config/index.ts';

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
    bodyType: RigidBodyType.Dynamic,
    density: 10_000,
    angularDamping: 0.1,
    linearDamping: 0.1, // Will be overwritten by caliber-specific value
    collisionEvent: ActiveEvents.CONTACT_FORCE_EVENTS,
    belongsCollisionGroup: CollisionGroup.BULLET,
    interactsCollisionGroup: CollisionGroup.ALL & ~CollisionGroup.TANK_TURRET_GUN_PARTS,
};
const defaultOptionsBulletRR = structuredClone(optionsBulletRR);
const tmpSpeed = vec2.create();

export function createBullet(options: Partial<Options> & {
    calibre: BulletCaliber,
    playerId: number,
    teamId: number
}, { world } = GameDI) {
    Object.assign(optionsBulletRR, defaultOptionsBulletRR);
    Object.assign(optionsBulletRR, options);

    const bulletCaliber = mapBulletCaliber[options.calibre];
    const speed = min(bulletCaliber.speed, MAX_BULLET_SPEED);

    if (isNumber(speed) && speed > 0) {
        tmpSpeed[0] = 0;
        tmpSpeed[1] = -speed;
        applyRotationToVector(tmpSpeed, tmpSpeed, options.rotation ?? 0);
        optionsBulletRR.speedX = tmpSpeed[0];
        optionsBulletRR.speedY = tmpSpeed[1];
    }
    optionsBulletRR.density = bulletCaliber.density;
    optionsBulletRR.linearDamping = bulletCaliber.linearDamping;

    const [bulletId] = createRectangleRR(optionsBulletRR);
    Bullet.addComponent(world, bulletId, options.calibre);
    TeamRef.addComponent(world, bulletId, options.teamId);
    PlayerRef.addComponent(world, bulletId, options.playerId);
    Hitable.addComponent(world, bulletId, min(bulletCaliber.width, bulletCaliber.height) / 10);
    Damagable.addComponent(world, bulletId, bulletCaliber.damage);
    DestroyBySpeed.addComponent(world, bulletId, MIN_BULLET_SPEED);

    return bulletId;
}

const optionsSpawnBullet = {
    x: 0,
    y: 0,
    width: 0,
    height: 0,
    color: new Float32Array(4).fill(1),
    calibre: BulletCaliber.Light as BulletCaliber,
    rotation: 0,
    playerId: 0,
    teamId: 0,
};
const tmpMatrix = mat4.create();
const tmpPosition = vec3.create() as Float32Array;

export function spawnBullet(vehicleEid: number) {
    const turretEid = Tank.turretEId[vehicleEid];
    const globalTransform = GlobalTransform.matrix.getBatch(turretEid);
    const bulletPosition = Firearms.bulletStartPosition.getBatch(turretEid);
    const bulletCaliber = mapBulletCaliber[Firearms.caliber[turretEid] as BulletCaliber];

    tmpPosition.set(bulletPosition);
    mat4.identity(tmpMatrix);
    mat4.translate(tmpMatrix, tmpMatrix, tmpPosition);
    mat4.multiply(tmpMatrix, globalTransform, tmpMatrix);

    // Dark color for bullets (30% of original brightness)
    Color.applyColorToArray(vehicleEid, optionsSpawnBullet.color);
    optionsSpawnBullet.color[0] *= 0.3;
    optionsSpawnBullet.color[1] *= 0.3;
    optionsSpawnBullet.color[2] *= 0.3;

    optionsSpawnBullet.x = getMatrixTranslationX(tmpMatrix);
    optionsSpawnBullet.y = getMatrixTranslationY(tmpMatrix);
    optionsSpawnBullet.width = bulletCaliber.width;
    optionsSpawnBullet.height = bulletCaliber.height;
    optionsSpawnBullet.rotation = getMatrixRotationZ(tmpMatrix);
    optionsSpawnBullet.calibre = Firearms.caliber[turretEid] as BulletCaliber;
    optionsSpawnBullet.teamId = TeamRef.id[vehicleEid];
    optionsSpawnBullet.playerId = PlayerRef.id[vehicleEid];

    createBullet(optionsSpawnBullet);

    // Spawn muzzle flash effect
    spawnMuzzleFlash({
        x: optionsSpawnBullet.x,
        y: optionsSpawnBullet.y,
        size: bulletCaliber.width * ExplosionConfig.muzzleFlashSizeMult,
        duration: ExplosionConfig.muzzleFlashDuration,
        rotation: optionsSpawnBullet.rotation,
    });

    const soundVolume = SoundConfig.shootBaseVolume + (bulletCaliber.width * SoundConfig.shootVolumePerWidth);
    spawnSoundAtPosition({
        type: SoundType.TankShoot,
        x: optionsSpawnBullet.x,
        y: optionsSpawnBullet.y,
        volume: soundVolume,
        destroyOnFinish: true,
    });
}