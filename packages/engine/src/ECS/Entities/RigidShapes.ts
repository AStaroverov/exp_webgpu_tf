import {
  createRectangle,
  createSphere,
} from "../../../../renderer3d_2/src/ECS/Entities/Shapes.ts";
import type { TColor } from "../../../../renderer3d_2/src/ECS/Components/Common.ts";
import type { EngineWorld } from "../createEngineWorld.ts";
import { getEngineComponents } from "../createEngineWorld.ts";
import type { PhysicalWorld } from "../../Physics/initPhysicalWorld.ts";
import { createBody } from "../../Physics/createBody.ts";
import { createRigidBox as createRigidBoxCollider, createRigidBall } from "../../Physics/createRigid.ts";

// Generic "give an entity a body + a render shape". Order: create the render
// entity (Transform/Shape/Color/Height), create the physics body+collider, then
// attach the bridge + state components. Returns [eid, pid].
//
// The crucial invariant every factory establishes: Height (render) = the body's
// FULL Z extent, so the §4 sync's baseZ = centerZ − Height/2 offset is exact for
// every kind. The render `z` is passed as the BOTTOM (center − halfHeight) so the
// initial render matches the physics centroid before the first sync.

// Box: render = a rectangle footprint extruded by `sz`; physics = cuboid(hx,hy,hz).
export function createRigidBox(
  world: EngineWorld,
  physicalWorld: PhysicalWorld,
  {
    x,
    y,
    z,
    sx,
    sy,
    sz,
    color,
  }: { x: number; y: number; z: number; sx: number; sy: number; sz: number; color: TColor },
): [number, number] {
  const { RigidBodyRef, RigidBodyState } = getEngineComponents(world);
  const hx = sx / 2;
  const hy = sy / 2;
  const hz = sz / 2;
  // render: footprint sx×sy in XY, extruded sz along Z; z passed is the BOTTOM.
  // createRectangle sets Height = sz via `depth`.
  const eid = createRectangle(world, {
    x,
    y,
    z: z - hz,
    width: sx,
    height: sy,
    color,
    depth: sz,
  });
  const body = createBody(physicalWorld, { type: "dynamic", x, y, z });
  const pid = createRigidBoxCollider(physicalWorld, body, hx, hy, hz);
  RigidBodyRef.addComponent(world, eid, pid);
  RigidBodyState.addComponent(world, eid);
  return [eid, pid];
}

// Sphere: render = createSphere (sets Height = 2r); physics = ball(r). Rotation-invariant.
export function createRigidSphere(
  world: EngineWorld,
  physicalWorld: PhysicalWorld,
  { x, y, z, radius, color }: { x: number; y: number; z: number; radius: number; color: TColor },
): [number, number] {
  const { RigidBodyRef, RigidBodyState } = getEngineComponents(world);
  const eid = createSphere(world, { x, y, z: z - radius, radius, color }); // z = bottom = center − r
  const body = createBody(physicalWorld, { type: "dynamic", x, y, z });
  const pid = createRigidBall(physicalWorld, body, radius);
  RigidBodyRef.addComponent(world, eid, pid);
  RigidBodyState.addComponent(world, eid);
  return [eid, pid];
}

// Ground: a FIXED, thin, wide box. Render = a large flat rectangle; physics = thin cuboid.
export function createGround(
  world: EngineWorld,
  physicalWorld: PhysicalWorld,
  {
    size = 200,
    thickness = 1,
    z = 0,
    color,
  }: { size?: number; thickness?: number; z?: number; color: TColor },
): [number, number] {
  const { RigidBodyRef, RigidBodyState } = getEngineComponents(world);
  const hz = thickness / 2;
  // The fixed body's center sits at z − hz; its render bottom is therefore
  // (z − hz) − hz = z − thickness. The top face lands exactly at z.
  const eid = createRectangle(world, {
    x: 0,
    y: 0,
    z: z - thickness,
    width: size,
    height: size,
    color,
    depth: thickness,
  });
  const body = createBody(physicalWorld, { type: "fixed", x: 0, y: 0, z: z - hz });
  const pid = createRigidBoxCollider(physicalWorld, body, size / 2, size / 2, hz);
  RigidBodyRef.addComponent(world, eid, pid);
  // Fixed bodies never wake, so this stays zero-synced; the identity quaternion +
  // initial render placement already match — harmless.
  RigidBodyState.addComponent(world, eid);
  return [eid, pid];
}
