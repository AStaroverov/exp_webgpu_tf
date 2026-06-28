import { createRectangle, createSphere } from "../../../../renderer/src/ECS/Entities/Shapes.ts";
import type { TColor } from "../../../../renderer/src/ECS/Components/Common.ts";
import type { EngineWorld } from "../createEngineWorld.ts";
import { createEntityId, getEngineComponents } from "../createEngineWorld.ts";

// MAIN half of the RigidShapes split (plan §6.1/§6.3). The worker owns Rapier, so each
// factory here only: (a) pulls eid from the shared monotonic counter, (b) builds the
// RENDER entity at that eid, (c) gives it RigidBodyState WITH the body spec — that single
// component-add seeds the spawn pose AND emits the SPAWN_BODY op for the worker (the op
// write lives INSIDE addComponent, not a separate method call). Returns the eid; there is
// no pid on main (it is worker-local).
//
// Center-origin: the render `z` and the physics body CENTER are the SAME value (the SDF is
// symmetric about local z=0), so each factory passes the same z to both the shape and the
// body spec. The Z extent rides in the shape's values (createRectangle's `depth`, the
// sphere's radius) — there is no separate Height component.

// Box: render = a rectangle footprint extruded by `sz`; physics = cuboid(hx,hy,hz).
export function createRigidBox(
  world: EngineWorld,
  {
    x,
    y,
    z,
    sx,
    sy,
    sz,
    color,
  }: { x: number; y: number; z: number; sx: number; sy: number; sz: number; color: TColor },
): number {
  const { RigidBodyState } = getEngineComponents(world);
  const hx = sx / 2;
  const hy = sy / 2;
  const hz = sz / 2;
  // eid from the shared monotonic counter (plan §4.2): the one authority the worker
  // adopts. createRectangle adopts this exact id rather than auto-allocating.
  const eid = createEntityId(world);
  // render: footprint sx×sy in XY, extruded sz along Z; z is the CENTER (== body center).
  createRectangle(world, { x, y, z, width: sx, height: sy, color, depth: sz, eid });
  // Spec position = body CENTER (z), matching the pre-split createBody translation.
  RigidBodyState.addComponent(world, eid, {
    kind: "box",
    bodyType: "dynamic",
    position: { x, y, z },
    halfExtents: { x: hx, y: hy, z: hz },
  });
  return eid;
}

// Sphere: render = createSphere; physics = ball(r). Rotation-invariant.
export function createRigidSphere(
  world: EngineWorld,
  { x, y, z, radius, color }: { x: number; y: number; z: number; radius: number; color: TColor },
): number {
  const { RigidBodyState } = getEngineComponents(world);
  const eid = createEntityId(world); // shared-counter authority (plan §4.2)
  // z = CENTER (== body center).
  createSphere(world, { x, y, z, radius, color, eid });
  RigidBodyState.addComponent(world, eid, {
    kind: "sphere",
    bodyType: "dynamic",
    position: { x, y, z },
    radius,
  });
  return eid;
}

// Ground: a FIXED, thin, wide box. Render = a large flat rectangle; physics = thin cuboid.
export function createGround(
  world: EngineWorld,
  {
    size = 200,
    thickness = 1,
    z = 0,
    color,
  }: { size?: number; thickness?: number; z?: number; color: TColor },
): number {
  const { RigidBodyState } = getEngineComponents(world);
  const hz = thickness / 2;
  const eid = createEntityId(world); // shared-counter authority (plan §4.2)
  // Center-origin: the slab center sits at z − hz so its top face lands exactly at z.
  // Render center == fixed-body center == z − hz.
  createRectangle(world, {
    x: 0,
    y: 0,
    z: z - hz,
    width: size,
    height: size,
    color,
    depth: thickness,
    eid,
  });
  RigidBodyState.addComponent(world, eid, {
    kind: "groundBox",
    bodyType: "fixed",
    position: { x: 0, y: 0, z: z - hz },
    halfExtents: { x: size / 2, y: size / 2, z: hz },
  });
  return eid;
}
