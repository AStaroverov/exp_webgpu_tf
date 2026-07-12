import { mat4 } from "gl-matrix";
import { createEngine } from "../../../../engine/src/createEngine.ts";
import {
  createEntityId,
  getEngineComponents,
  type EngineWorld,
} from "../../../../engine/src/ECS/createEngineWorld.ts";
import { createGround } from "../../../../engine/src/ECS/Entities/RigidShapes.ts";
import { addTransformComponents } from "../../../../renderer/src/ECS/Components/Transform.ts";
import { SunLight } from "../../../../renderer/src/ECS/Systems/SunLight.ts";
import {
  cameraAzimuth,
  cameraElevation,
  cameraZoom,
  setCameraPosition,
} from "../../../../renderer/src/ECS/Systems/ResizeSystem.ts";
import { buildUnit, type UnitInstance } from "../../../src/Entities/unit.ts";
import { buildSwordsman } from "../../../src/Entities/swordsman.ts";
import { buildLightsaber } from "../../../src/Entities/lightsaber.ts";
import { createWanderComponent, createWanderSystem } from "./wander.ts";

// Proto scene: 3 units, each backed by a dynamic sphere collider, wandering to random
// points inside a fixed radius around the origin.
//
// PHYSICS vs VISUAL split: the unit (Entities/unit.ts) is a render-only hierarchy whose
// root LocalTransform is rewritten every frame by its own animation — so it can't BE the
// physics entity. Instead each unit gets:
//   - a bare collider body (RigidBodyState + Velocity + Wander), invisible, simulated;
//   - a carrier entity placed each frame from the body's physics XY (+ a facing yaw);
//   - the unit root parented under the carrier, animating locally on top.
// The wander system steers Velocity → SET_VELOCITY op → worker setLinvel; the worker's
// pose flows back into the body's RigidBodyState, which the carrier follows.

const SCALE = 1;
const COLLIDER_RADIUS = 0.6 * SCALE;
const WANDER_RADIUS = 7;
const UNIT_COUNT = 3;
const GROUND_Z = 0; // carrier sits on the ground plane; the unit's own HOVER lifts it
const FORWARD_OFFSET = Math.PI / 2; // unit model faces +Y; turn it toward travel

type Unit = { bodyEid: number; carrier: number; instance: UnitInstance; yaw: number; hurt: number };

const UNIT_COLOR: [number, number, number] = [0.13, 0.34, 0.56];
const HURT_COLOR: [number, number, number] = [0.9, 0.15, 0.1];

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement;

  const engine = await createEngine({ canvas });
  const world = engine.world as EngineWorld;
  const { Children, RigidBodyState, Velocity, LocalTransform, ShapeCaster, Color } =
    getEngineComponents(world);

  const Wander = createWanderComponent(world);
  const wander = createWanderSystem(world, Wander);

  // Sun + camera (tilted top-down framing the wander disk).
  SunLight.enabled = true;
  SunLight.angle = 2.4;
  SunLight.elevation = 0.95;
  SunLight.intensity = 0.9;
  SunLight.color = [1.0, 0.93, 0.82];

  setCameraPosition(0, 0);
  cameraZoom.value = 12;
  cameraElevation.value = 55;
  cameraAzimuth.value = 45;

  createGround(world, { size: 40, thickness: 1, z: 0, color: [0.18, 0.2, 0.24, 1] });

  const units: Unit[] = [];
  for (let i = 0; i < UNIT_COUNT; i++) {
    const angle = (i / UNIT_COUNT) * Math.PI * 2;
    const x = Math.cos(angle) * WANDER_RADIUS * 0.5;
    const y = Math.sin(angle) * WANDER_RADIUS * 0.5;

    const bodyEid = createEntityId(world);
    RigidBodyState.addComponent(world, bodyEid, {
      kind: "sphere",
      bodyType: "dynamic",
      position: { x, y, z: COLLIDER_RADIUS },
      radius: COLLIDER_RADIUS,
    });
    Velocity.addComponent(world, bodyEid);
    Wander.addComponent(world, bodyEid, {
      centerX: 0,
      centerY: 0,
      radius: WANDER_RADIUS,
      speed: 3,
      arrive: 0.4,
    });

    const carrier = createEntityId(world);
    addTransformComponents(world, carrier);
    Children.addComponent(world, carrier);
    Children.addChild(engine.sceneRoot, carrier);

    const instance = buildUnit(world, { scale: SCALE });
    Children.addChild(carrier, instance.root);

    units.push({ bodyEid, carrier, instance, yaw: 0, hurt: 0 });
  }
  const unitByBodyEid = new Map(units.map((u) => [u.bodyEid, u]));

  // A swordsman at the origin, endlessly swinging. The lightsaber factory already
  // attached a ShapeCaster along the blade; wanderers straying into the swing flash red.
  const swordsman = buildSwordsman(world, {
    scale: SCALE,
    parts: { unit: buildUnit, sword: buildLightsaber },
  });
  Children.addChild(engine.sceneRoot, swordsman.root);
  const swordRoot = swordsman.bones["sword/root"];

  function applyBladeHits(delta: number): void {
    const hitCount = ShapeCaster.getHitCount(swordRoot);
    for (let i = 0; i < hitCount; i++) {
      const victim = unitByBodyEid.get(ShapeCaster.getHit(swordRoot, i));
      if (victim) victim.hurt = 1;
    }
    for (let i = 0; i < units.length; i++) {
      const u = units[i];
      if (u.hurt === 0) continue;
      u.hurt = Math.max(0, u.hurt - delta * 2);
      const t = u.hurt;
      Color.set$(
        u.instance.bones.body,
        UNIT_COLOR[0] + (HURT_COLOR[0] - UNIT_COLOR[0]) * t,
        UNIT_COLOR[1] + (HURT_COLOR[1] - UNIT_COLOR[1]) * t,
        UNIT_COLOR[2] + (HURT_COLOR[2] - UNIT_COLOR[2]) * t,
        1,
      );
    }
  }

  function placeUnit(u: Unit, delta: number): void {
    const x = RigidBodyState.position.get(u.bodyEid, 0);
    const y = RigidBodyState.position.get(u.bodyEid, 1);
    const vx = Velocity.x.get(u.bodyEid);
    const vy = Velocity.y.get(u.bodyEid);
    const moving = Math.hypot(vx, vy) > 0.05;
    if (moving) u.yaw = Math.atan2(vy, vx) - FORWARD_OFFSET;

    const m = LocalTransform.matrix.getBatch(u.carrier);
    mat4.identity(m);
    mat4.translate(m, m, [x, y, GROUND_Z]);
    mat4.rotateZ(m, m, u.yaw);

    (moving ? u.instance.animations.movement : u.instance.animations.idle)(delta);
  }

  let then = performance.now();
  function loop(now: number): void {
    const delta = Math.min(now - then, 16.6667) / 1000;
    then = now;

    wander();
    for (let i = 0; i < units.length; i++) placeUnit(units[i], delta);
    swordsman.animations.sword_slice(delta);
    applyBladeHits(delta);
    engine.tick(delta);

    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="color:#f88;padding:20px">${(err as Error)?.stack ?? err}</pre>`;
  console.error(err);
});
