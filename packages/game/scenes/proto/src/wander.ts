import { addComponent, query, World } from "bitecs";
import { defineComponent } from "../../../../common/src/component.ts";
import {
  getEngineComponents,
  type EngineWorld,
} from "../../../../engine/src/ECS/createEngineWorld.ts";

// Wander: roam to random points inside a fixed disk around a center, picking a fresh target
// on arrival. Pure data — the steering lives in createWanderSystem, which queries
// [Wander, RigidBodyState, Velocity] and writes Velocity toward the current target.
export type WanderConfig = {
  centerX: number;
  centerY: number;
  radius: number;
  speed: number;
  arrive: number;
};

export const createWanderComponent = defineComponent((Wander, ctx) => {
  const centerX = ctx.table.flat(Float64Array);
  const centerY = ctx.table.flat(Float64Array);
  const radius = ctx.table.flat(Float64Array);
  const speed = ctx.table.flat(Float64Array);
  const arrive = ctx.table.flat(Float64Array);
  const targetX = ctx.table.flat(Float64Array);
  const targetY = ctx.table.flat(Float64Array);
  const hasTarget = ctx.table.flat(Int8Array);
  return {
    centerX,
    centerY,
    radius,
    speed,
    arrive,
    targetX,
    targetY,
    hasTarget,
    addComponent(world: World, eid: number, cfg: WanderConfig) {
      addComponent(world, eid, Wander);
      centerX.set(eid, cfg.centerX);
      centerY.set(eid, cfg.centerY);
      radius.set(eid, cfg.radius);
      speed.set(eid, cfg.speed);
      arrive.set(eid, cfg.arrive);
      hasTarget.set(eid, 0);
    },
  };
});

export type WanderComponent = ReturnType<typeof createWanderComponent>;

export function createWanderSystem(world: EngineWorld, Wander: WanderComponent): () => void {
  const { RigidBodyState, Velocity } = getEngineComponents(world);

  function pickTarget(eid: number): void {
    // uniform point in the disk: r = radius·√u so area, not radius, is uniform
    const angle = Math.random() * Math.PI * 2;
    const r = Math.sqrt(Math.random()) * Wander.radius.get(eid);
    Wander.targetX.set(eid, Wander.centerX.get(eid) + Math.cos(angle) * r);
    Wander.targetY.set(eid, Wander.centerY.get(eid) + Math.sin(angle) * r);
    Wander.hasTarget.set(eid, 1);
  }

  return function wander() {
    const entities = query(world, [Wander, RigidBodyState, Velocity]);
    for (let i = 0; i < entities.length; i++) {
      const eid = entities[i];
      if (Wander.hasTarget.get(eid) === 0) pickTarget(eid);

      const px = RigidBodyState.position.get(eid, 0);
      const py = RigidBodyState.position.get(eid, 1);
      const dx = Wander.targetX.get(eid) - px;
      const dy = Wander.targetY.get(eid) - py;
      const dist = Math.hypot(dx, dy);

      if (dist <= Wander.arrive.get(eid)) {
        pickTarget(eid);
        Velocity.set(eid, 0, 0, 0);
        continue;
      }

      const speed = Wander.speed.get(eid);
      Velocity.set(eid, (dx / dist) * speed, (dy / dist) * speed, 0);
    }
  };
}
