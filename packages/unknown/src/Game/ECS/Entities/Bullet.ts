import { GameDI } from "../../DI/GameDI.ts";
import { RenderDI } from "../../DI/RenderDI.ts";
import { isNumber } from "lodash-es";
import { applyRotationToVector } from "../../Physical/applyRotationToVector.ts";
import { createRectangleRR } from "../Components/RigidRender.ts";
import { spawnMuzzleFlash } from "./MuzzleFlash.ts";
import { SoundType } from "../Components/Sound.ts";
import { spawnSoundAtPosition } from "./Sound.ts";
import { mat4, vec2, vec3 } from "gl-matrix";
import {
  getMatrixRotationZ,
  getMatrixTranslationX,
  getMatrixTranslationY,
  GlobalTransform,
} from "../../../../../renderer/src/ECS/Components/Transform.ts";
import { ZIndex } from "../../consts.ts";
import { ActiveEvents, RigidBodyType } from "@dimforge/rapier2d-simd";
import { CollisionGroup } from "../../Physical/createRigid.ts";
import { BulletCaliber, mapBulletCaliber, MAX_BULLET_SPEED } from "../Components/Bullet.ts";
import { getGameComponents } from "../createGameWorld.ts";
import { min, PI } from "../../../../../../lib/math.ts";
import { ExplosionConfig, SoundConfig } from "../../Config/index.ts";

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
  linearDamping: 0.1,
  collisionEvent: ActiveEvents.CONTACT_FORCE_EVENTS,
  belongsCollisionGroup: CollisionGroup.BULLET,
  interactsCollisionGroup:
    CollisionGroup.ALL & ~CollisionGroup.TANK_TURRET_GUN_PARTS & ~CollisionGroup.BULLET,
};
const defaultOptionsBulletRR = structuredClone(optionsBulletRR);
const tmpSpeed = vec2.create();

export function createBullet(
  options: Partial<Options> & {
    calibre: BulletCaliber;
    playerId: number;
    teamId: number;
  },
  { world } = GameDI,
) {
  const {
    Bullet,
    TeamRef,
    PlayerRef,
    Hitable,
    Damagable,
    DestroyByDistance,
    Explodable,
    Color,
    LightEmitter,
  } = getGameComponents(world);

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
  optionsBulletRR.angularSpeed = bulletCaliber.angularSpeed ?? 0;

  const [bulletId] = createRectangleRR(optionsBulletRR);
  Bullet.addComponent(world, bulletId, options.calibre);
  TeamRef.addComponent(world, bulletId, options.teamId);
  PlayerRef.addComponent(world, bulletId, options.playerId);
  Hitable.addComponent(world, bulletId, bulletCaliber.health);
  Damagable.addComponent(world, bulletId, bulletCaliber.damage);
  DestroyByDistance.addComponent(
    world,
    bulletId,
    optionsBulletRR.x,
    optionsBulletRR.y,
    bulletCaliber.maxDistance,
  );

  if (bulletCaliber.explosion) {
    Explodable.addComponent(world, bulletId, bulletCaliber.explosion);
  }

  // Glowing projectile: override the dimmed vehicle color and feed the RC emission
  // pass — the bullet's own SDF rect is the emitter silhouette.
  if (bulletCaliber.light && RenderDI.enabled) {
    Color.set$(bulletId, ...bulletCaliber.light.color, 1);
    LightEmitter.addComponent(world, bulletId, bulletCaliber.light.intensity);
  }

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
const tmpSpawnDelta = new Float32Array(2);

export function spawnBullet(vehicleEid: number, { world } = GameDI) {
  const { Tank, Firearms, SpawnDeltaPosition, TeamRef, PlayerRef, Color } =
    getGameComponents(world);

  const turretEid = Tank.turretEId.get(vehicleEid);
  const globalTransform = GlobalTransform.matrix.getBatch(turretEid);
  const bulletPosition = SpawnDeltaPosition.position.getBatch(turretEid, tmpSpawnDelta);
  const bulletCaliber = mapBulletCaliber[Firearms.caliber.get(turretEid) as BulletCaliber];

  tmpPosition.set(bulletPosition);
  mat4.identity(tmpMatrix);
  mat4.translate(tmpMatrix, tmpMatrix, tmpPosition);
  mat4.multiply(tmpMatrix, globalTransform, tmpMatrix);

  Color.applyColorToArray(vehicleEid, optionsSpawnBullet.color);
  optionsSpawnBullet.color[0] *= 0.3;
  optionsSpawnBullet.color[1] *= 0.3;
  optionsSpawnBullet.color[2] *= 0.3;

  optionsSpawnBullet.x = getMatrixTranslationX(tmpMatrix);
  optionsSpawnBullet.y = getMatrixTranslationY(tmpMatrix);
  optionsSpawnBullet.width = bulletCaliber.width;
  optionsSpawnBullet.height = bulletCaliber.height;
  optionsSpawnBullet.rotation = getMatrixRotationZ(tmpMatrix);
  optionsSpawnBullet.calibre = Firearms.caliber.get(turretEid) as BulletCaliber;
  optionsSpawnBullet.teamId = TeamRef.id.get(vehicleEid);
  optionsSpawnBullet.playerId = PlayerRef.id.get(vehicleEid);

  createBullet(optionsSpawnBullet);

  spawnMuzzleFlash({
    x: optionsSpawnBullet.x,
    y: optionsSpawnBullet.y,
    size: bulletCaliber.height * ExplosionConfig.muzzleFlashSizeMult,
    duration: ExplosionConfig.muzzleFlashDuration,
    rotation: optionsSpawnBullet.rotation + PI / 2,
  });

  const soundVolume =
    SoundConfig.shootBaseVolume + bulletCaliber.width * SoundConfig.shootVolumePerWidth;
  spawnSoundAtPosition({
    type: SoundType.TankShoot,
    x: optionsSpawnBullet.x,
    y: optionsSpawnBullet.y,
    volume: soundVolume,
    destroyOnFinish: true,
  });
}
