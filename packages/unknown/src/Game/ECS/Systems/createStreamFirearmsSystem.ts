import { addEntity, hasComponent, query } from "bitecs";
import { ActiveEvents } from "@dimforge/rapier2d-simd";
import { GameDI } from "../../DI/GameDI.ts";
import { RenderDI } from "../../DI/RenderDI.ts";
import { getGameComponents } from "../createGameWorld.ts";
import { CollisionGroup, createRigidCircle } from "../../Physical/createRigid.ts";
import { DamageKind } from "../Components/Damagable.ts";
import { VFXType } from "../Components/VFX.ts";
import {
  ALL_VEHICLE_PARTS_MASK,
  StreamCaliberConfig,
  StreamParticleLightConfig,
} from "../../Config/index.ts";
import { ShapeKind } from "../../../../../renderer/src/ECS/Components/Shape.ts";
import {
  addTransformComponents,
  applyMatrixTranslate,
  getMatrixRotationZ,
  getMatrixTranslationX,
  getMatrixTranslationY,
  GlobalTransform,
  LocalTransform,
} from "../../../../../renderer/src/ECS/Components/Transform.ts";
import { seededRandomRangeFloat } from "../../../../../../lib/random.ts";
import { cos, PI, sin } from "../../../../../../lib/math.ts";
import { ZIndex } from "../../consts.ts";

type RigidOptions = Parameters<typeof createRigidCircle>[0];
const optionsParticleRigid: RigidOptions = {
  x: 0,
  y: 0,
  rotation: 0,
  speedX: 0,
  speedY: 0,
  radius: 0,
  density: 1,
  gravityScale: 0,
  linearDamping: 0,
  angularDamping: 0,
  sensor: true,
  collisionEvent: ActiveEvents.COLLISION_EVENTS,
  belongsCollisionGroup: CollisionGroup.PARTICLE,
  interactsCollisionGroup: ALL_VEHICLE_PARTS_MASK | CollisionGroup.OBSTACLE,
};

/**
 * Sprays sensor particles from every `StreamFirearms` turret while its shoot
 * flag is held and the magazine isn't spent (`fireDurationMs` of emission →
 * `reloadMs` pause). Disjoint from the bullet spawner by component (`StreamFirearms`
 * vs `Firearms`). Each particle is composed inline: the gameplay pieces
 * (sensor body + `Damagable`/`Dotable`/`SensorHits` + `DestroyByTimeout`) are added unconditionally
 * so the weapon works headless; the visual pieces only when rendering is on.
 */
export function createStreamFirearmsSystem({ world } = GameDI) {
  const {
    VehicleTurret,
    TurretController,
    StreamFirearms,
    SpawnDeltaPosition,
    Damagable,
    Dotable,
    SensorHits,
    DestroyByTimeout,
    RigidBodyRef,
    RigidBodyState,
    PlayerRef,
    TeamRef,
    Parent,
    VFX,
    Progress,
    Wander,
    Shape,
    Color,
    LightEmitter,
    Stunned,
    Vehicle,
  } = getGameComponents(world);

  const emitBurst = (turretEid: number) => {
    const cfg = StreamCaliberConfig[StreamFirearms.caliberRef[turretEid]];
    const matrix = GlobalTransform.matrix.getBatch(turretEid);
    const facing = getMatrixRotationZ(matrix);
    // Muzzle world position: turret transform × the local spawn delta (gun tip).
    const dx = SpawnDeltaPosition.position.get(turretEid, 0);
    const dy = SpawnDeltaPosition.position.get(turretEid, 1);
    const x = getMatrixTranslationX(matrix) + cos(facing) * dx - sin(facing) * dy;
    const y = getMatrixTranslationY(matrix) + sin(facing) * dx + cos(facing) * dy;
    const vehicleEid = Parent.id[turretEid];
    const isFire = cfg.kind === DamageKind.Fire;
    const lifetimeMs = cfg.lifetimeMs;

    for (let i = 0; i < cfg.count; i++) {
      // Seeded RNG, not Math.random — spread must be reproducible in training.
      const angle = facing + seededRandomRangeFloat(-cfg.spreadRad, cfg.spreadRad);
      const speed = cfg.speed * seededRandomRangeFloat(0.85, 1.15);

      optionsParticleRigid.x = x;
      optionsParticleRigid.y = y;
      optionsParticleRigid.rotation = angle;
      optionsParticleRigid.speedX = cos(angle) * speed;
      optionsParticleRigid.speedY = sin(angle) * speed;
      optionsParticleRigid.radius = cfg.particleRadius * 2; // createRigidCircle halves it
      optionsParticleRigid.linearDamping = cfg.linearDamping;

      const particleEid = addEntity(world);
      addTransformComponents(world, particleEid);
      applyMatrixTranslate(LocalTransform.matrix.getBatch(particleEid), x, y, ZIndex.Explosion);

      // Gameplay — always, so sensor overlaps + status application work headless.
      const physicalId = createRigidCircle(optionsParticleRigid);
      RigidBodyRef.addComponent(world, particleEid, physicalId);
      RigidBodyState.addComponent(world, particleEid);
      Damagable.addComponent(world, particleEid, cfg.damage, cfg.kind);
      Dotable.addComponent(world, particleEid, cfg.dot.dps, cfg.dot.durationMs, cfg.kind);
      SensorHits.addComponent(world, particleEid, cfg.hitLifeCostMs);
      DestroyByTimeout.addComponent(world, particleEid, lifetimeMs);
      // Seeded phase → reproducible meander; frequency converted Hz → rad/ms.
      Wander.addComponent(
        world,
        particleEid,
        seededRandomRangeFloat(0, PI * 2),
        (cfg.wander.frequency * PI * 2) / 1000,
        cfg.wander.angularSpeed,
      );
      TeamRef.addComponent(world, particleEid, TeamRef.id[vehicleEid]);
      PlayerRef.addComponent(world, particleEid, PlayerRef.id[vehicleEid]);

      // Visual — the look comes from the VFXType shader, not Color.
      if (RenderDI.enabled) {
        VFX.addComponent(world, particleEid, isFire ? VFXType.Flame : VFXType.Frost);
        Progress.addComponent(world, particleEid, lifetimeMs);

        // Travelling glow: an alpha-0 SDF circle feeds the emission pass
        // (invisible in the main pass; light emitters never cast shadows).
        const light = isFire ? StreamParticleLightConfig.flame : StreamParticleLightConfig.frost;
        Shape.addComponent(world, particleEid, ShapeKind.Circle, light.radius);
        Color.addComponent(world, particleEid, light.color[0], light.color[1], light.color[2], 0);
        LightEmitter.addComponent(world, particleEid, light.intensity);
      }
    }
  };

  return (delta: number) => {
    const turretEids = query(world, [VehicleTurret, TurretController, StreamFirearms]);

    for (let i = 0; i < turretEids.length; i++) {
      const turretEid = turretEids[i];
      const cfg = StreamCaliberConfig[StreamFirearms.caliberRef[turretEid]];
      const interval = cfg.emitIntervalMs;

      StreamFirearms.updateReloading(turretEid, delta);

      // Stun gate after updateReloading — reload keeps ticking through a stun.
      // Verify the parent is still a Vehicle: a stale/recycled eid must not gate the gun.
      const vehicleEid = Parent.id[turretEid];
      if (hasComponent(world, vehicleEid, Vehicle) && hasComponent(world, vehicleEid, Stunned)) {
        continue;
      }

      if (StreamFirearms.isReloading(turretEid) || !TurretController.shouldShoot(turretEid)) {
        // Primed reset: a fresh (post-reload) hold emits on its very first tick.
        StreamFirearms.emitAccMs[turretEid] = interval;
        continue;
      }

      // Magazine: `firedMs` of emission spent; releasing the trigger does NOT
      // refill it — only the reload does (see `startReloading`).
      StreamFirearms.firedMs[turretEid] += delta;
      if (StreamFirearms.firedMs[turretEid] >= cfg.fireDurationMs) {
        StreamFirearms.startReloading(turretEid, cfg.reloadMs);
        continue;
      }

      StreamFirearms.emitAccMs[turretEid] += delta;

      while (StreamFirearms.emitAccMs[turretEid] >= interval) {
        StreamFirearms.emitAccMs[turretEid] -= interval;
        emitBurst(turretEid);
      }
    }
  };
}
