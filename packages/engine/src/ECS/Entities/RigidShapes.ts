import { createRectangle, createSphere } from "../../../../renderer3d_2/src/ECS/Entities/Shapes.ts";
import type { TColor } from "../../../../renderer3d_2/src/ECS/Components/Common.ts";
import type { EngineWorld } from "../createEngineWorld.ts";
import { getEngineComponents, getEngineSab } from "../createEngineWorld.ts";
import { spawnBox, spawnGroundBox, spawnSphere } from "../../Physics/opChannel.ts";
import { EngineDI } from "../../DI/EngineDI.ts";

// MAIN half of the RigidShapes split (plan §6.1/§6.3). The worker owns Rapier, so each
// factory here only: (a) pulls eid from the shared monotonic counter, (b) builds the
// RENDER entity at that eid, (c) adds RigidBodyState so the apply system's query matches
// and the shape shows at its spawn pose until the first worker publish, (d) posts a
// SPAWN_BODY op (carrying the body CENTER + dims) for the worker to materialize. Returns
// the eid; there is no pid on main (it is worker-local).
//
// The crucial invariant every factory establishes: Height (render) = the body's FULL Z
// extent, so the §4 sync's baseZ = centerZ − Height/2 offset is exact for every kind.
// The render `z` is passed as the BOTTOM (center − halfHeight); the op carries the
// physics CENTER (the same z createBody received before the split) so the worker mirrors
// the render placement exactly.

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
  // @TODO: инкапсулируй 41 строку в createEntityId
  const eid = getEngineSab(world).nextEid();
  // render: footprint sx×sy in XY, extruded sz along Z; z passed is the BOTTOM.
  // createRectangle sets Height = sz via `depth`.
  createRectangle(world, { x, y, z: z - hz, width: sx, height: sy, color, depth: sz, eid });
  RigidBodyState.addComponent(world, eid);
  // op position = body CENTER (z), matching the pre-split createBody translation.
  EngineDI.postOps([spawnBox(eid, "dynamic", { x, y, z }, hx, hy, hz)]);
  return eid;
}

// Sphere: render = createSphere (sets Height = 2r); physics = ball(r). Rotation-invariant.
export function createRigidSphere(
  world: EngineWorld,
  { x, y, z, radius, color }: { x: number; y: number; z: number; radius: number; color: TColor },
): number {
  const { RigidBodyState } = getEngineComponents(world);
  const eid = getEngineSab(world).nextEid(); // shared-counter authority (plan §4.2)
  // z = bottom = center − r
  createSphere(world, { x, y, z: z - radius, radius, color, eid });
  RigidBodyState.addComponent(world, eid);
  EngineDI.postOps([spawnSphere(eid, "dynamic", { x, y, z }, radius)]); // op pos = CENTER
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
  const eid = getEngineSab(world).nextEid(); // shared-counter authority (plan §4.2)
  // The fixed body's center sits at z − hz; its render bottom is therefore
  // (z − hz) − hz = z − thickness. The top face lands exactly at z.
  createRectangle(world, {
    x: 0,
    y: 0,
    z: z - thickness,
    width: size,
    height: size,
    color,
    depth: thickness,
    eid,
  });
  RigidBodyState.addComponent(world, eid);
  // op position = fixed body CENTER (z − hz), matching the pre-split createBody call.
  EngineDI.postOps([spawnGroundBox(eid, { x: 0, y: 0, z: z - hz }, size / 2, size / 2, hz)]);
  return eid;
}
