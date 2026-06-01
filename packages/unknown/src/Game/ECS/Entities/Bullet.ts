import { isNumber } from 'lodash-es';
import { applyRotationToVector } from '../../Physical/applyRotationToVector.ts';
import { spawnRectanglePart } from './spawnPart.ts';
import { getNodeByPhysics, getNodeRender, getTurretPhysOfHull } from '../refs.ts';
import { Worlds } from '../../DI/Worlds.ts';
import { getRenderWorldComponents } from '../createRenderWorld.ts';
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
import { ZIndex } from '../../consts.ts';
import { ActiveEvents, RigidBodyType } from '@dimforge/rapier2d-simd';
import { CollisionGroup } from '../../Physical/createRigid.ts';
import { BulletCaliber, mapBulletCaliber, MAX_BULLET_SPEED, MIN_BULLET_SPEED } from '../Components/Bullet.ts';
import { getPhysicsWorldComponents } from '../createPhysicsWorld.ts';
import { getBrainWorldComponents } from '../createBrainWorld.ts';
import { min, PI } from '../../../../../../lib/math.ts';
import { ExplosionConfig, SoundConfig } from '../../Config/index.ts';

type Options = Parameters<typeof spawnRectanglePart>[0];
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
    linearDamping: 0.1,
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
}, { physicsWorld: world } = Worlds) {
    const { Bullet, TeamRef, PlayerRef, Hitable, Damagable, DestroyBySpeed } = getPhysicsWorldComponents(world);

    Object.assign(optionsBulletRR, defaultOptionsBulletRR);
    Object.assign(optionsBulletRR, options);

    const bulletCaliber = mapBulletCaliber[options.calibre];
    const speed = min(bulletCaliber.speed, MAX_BULLET_SPEED);

    if (isNumber(speed) && speed > 0) {
        tmpSpeed[0] = speed;
        tmpSpeed[1] = 0;
        applyRotationToVector(tmpSpeed, tmpSpeed, options.rotation ?? 0);
        optionsBulletRR.speedX = tmpSpeed[0];
        optionsBulletRR.speedY = tmpSpeed[1];
    }
    optionsBulletRR.density = bulletCaliber.density;
    optionsBulletRR.linearDamping = bulletCaliber.linearDamping;

    const [bulletPhysEid] = spawnRectanglePart(optionsBulletRR);
    Bullet.addComponent(world, bulletPhysEid, options.calibre);
    // Cheap static team/player copy for the cold contact path (saveHitters).
    TeamRef.addComponent(world, bulletPhysEid, options.teamId);
    PlayerRef.addComponent(world, bulletPhysEid, options.playerId);
    Hitable.addComponent(world, bulletPhysEid, min(bulletCaliber.width, bulletCaliber.height) / 10);
    Damagable.addComponent(world, bulletPhysEid, bulletCaliber.damage);
    DestroyBySpeed.addComponent(world, bulletPhysEid, MIN_BULLET_SPEED);

    return bulletPhysEid;
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

export function spawnBullet(hullBrain: number, { renderWorld, brainWorld } = Worlds) {
    const { Firearms, TeamRef, PlayerRef } = getBrainWorldComponents(brainWorld);
    const { Color } = getRenderWorldComponents(renderWorld);

    // hull node -> turret child node (-> turret ATOM -> turret node).
    const turretPhysEid = getTurretPhysOfHull(hullBrain);
    const turretBrain = getNodeByPhysics(turretPhysEid);
    const turretRenderEid = getNodeRender(turretBrain);
    const vehicleRenderEid = getNodeRender(hullBrain);
    const globalTransform = GlobalTransform.matrix.getBatch(turretRenderEid);
    const bulletPosition = Firearms.bulletStartPosition.getBatch(turretBrain);
    const bulletCaliber = mapBulletCaliber[Firearms.caliber[turretBrain] as BulletCaliber];

    tmpPosition.set(bulletPosition);
    mat4.identity(tmpMatrix);
    mat4.translate(tmpMatrix, tmpMatrix, tmpPosition);
    mat4.multiply(tmpMatrix, globalTransform, tmpMatrix);

    Color.applyColorToArray(vehicleRenderEid, optionsSpawnBullet.color);
    optionsSpawnBullet.color[0] *= 0.3;
    optionsSpawnBullet.color[1] *= 0.3;
    optionsSpawnBullet.color[2] *= 0.3;

    optionsSpawnBullet.x = getMatrixTranslationX(tmpMatrix);
    optionsSpawnBullet.y = getMatrixTranslationY(tmpMatrix);
    optionsSpawnBullet.width = bulletCaliber.width;
    optionsSpawnBullet.height = bulletCaliber.height;
    optionsSpawnBullet.rotation = getMatrixRotationZ(tmpMatrix);
    optionsSpawnBullet.calibre = Firearms.caliber[turretBrain] as BulletCaliber;
    optionsSpawnBullet.teamId = TeamRef.id[hullBrain];
    optionsSpawnBullet.playerId = PlayerRef.id[hullBrain];

    createBullet(optionsSpawnBullet);

    spawnMuzzleFlash({
        x: optionsSpawnBullet.x,
        y: optionsSpawnBullet.y,
        size: bulletCaliber.height * ExplosionConfig.muzzleFlashSizeMult,
        duration: ExplosionConfig.muzzleFlashDuration,
        rotation: optionsSpawnBullet.rotation + PI / 2,
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
