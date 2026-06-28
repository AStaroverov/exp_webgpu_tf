import { createWorld as createBitecsWorld, createEntityIndex, World } from "bitecs";
import {
  createRenderComponents,
  type RenderComponents,
} from "../../../renderer3d_2/src/ECS/world.ts";
import { createEngineComponents, type EngineComponentsLocal } from "./components.ts";
import {
  createEngineSab,
  createEngineSabFromBundle,
  type EngineSab,
} from "../sab/engineSab.ts";
import type { SabBundle } from "../../../renderer3d_2/src/sab/registry.ts";

// The full component set on an engine world: renderer3d_2's render components
// (GlobalTransform, LocalTransform, Shape, Color, Height, LightEmitter, …) spread
// together with the physics-bridge components (RigidBodyRef, RigidBodyState).
// Because we spread createRenderComponents(world) into world.components, every
// renderer3d_2 system (createDrawShapeSystem, createTransformSystem,
// createVoxelSystem) finds exactly the components it reads via
// getRenderComponents — so this world is drop-in compatible with all of them.
export type EngineComponents = RenderComponents & ReturnType<typeof createEngineComponents>;

export type EngineWorld = World<{
  components: EngineComponents;
  // The shared bridge memory (DATA + CONTROL SABs, eid counter, pose banks).
  // Lives on the world so systems/entity factories reach it without a singleton.
  sab: EngineSab;
  time: {
    delta: number;
    elapsed: number;
    last: number;
  };
}>;

export function createEngineWorld(): EngineWorld {
  const sab = createEngineSab();
  const context = {
    components: null as unknown as EngineComponents,
    sab,
    time: {
      delta: 0,
      elapsed: 0,
      last: performance.now(),
    },
  };
  // versioning:false → raw eid === id, required by the shared-counter adoption
  // path (plan §4.2): eids come from sab.nextEid(), the world adopts them as-is.
  const world = createBitecsWorld(createEntityIndex(), context) as EngineWorld;
  context.components = {
    ...createRenderComponents(world),
    ...createEngineComponents(world),
  };
  return world;
}

// The PHYSICS-WORKER world: bitecs world + the SAME shared SAB (bound from the
// received bundle, not allocated) but ONLY the engine/physics components.
//
// Render components are SKIPPED ON PURPOSE: the worker never renders, so Shape /
// Color / LocalTransform / GlobalTransform / LightEmitter / Rope (~30MB) would be
// dead memory there. Crucially, those factories also register hidden obs-shadow
// components in bitflag order (plan §4.4) — registering them only on main and not in
// the worker is FINE precisely because we never compare entityMasks across threads
// (presence is mirrored via the op channel, not read from shared bytes). The bridge
// components (RigidBodyRef, RigidBodyState) ARE registered here so the worker can
// adopt eids, write pids, and produce pose into the shared banks.
export type PhysicsWorkerWorld = World<{
  components: EngineComponentsLocal;
  sab: EngineSab;
  time: {
    delta: number;
    elapsed: number;
    last: number;
  };
}>;

export function createPhysicsWorkerWorld(bundle: SabBundle): PhysicsWorkerWorld {
  const sab = createEngineSabFromBundle(bundle);
  const context = {
    components: null as unknown as EngineComponentsLocal,
    sab,
    time: {
      delta: 0,
      elapsed: 0,
      last: 0, // worker has no rAF clock; the self-clock loop advances this (plan §7)
    },
  };
  // versioning:false → raw eid === id, required by adoptEntity (plan §4.2): the worker
  // adopts the exact eid main pulled from the shared NEXT_EID counter.
  const world = createBitecsWorld(createEntityIndex(), context) as PhysicsWorkerWorld;
  context.components = createEngineComponents(world);
  return world;
}

// Getters accept either world flavor: the full main world (render + engine) or the
// worker world (engine only). Both carry the engine/physics components + the SAB,
// which is all the physics systems and entity factories read.
type EngineWorldLike = World<{
  components?: EngineComponents | EngineComponentsLocal;
  sab?: EngineSab;
}>;

// Return type is the FULL main-world set (render + engine): main-thread readers
// destructure render components (LightEmitter, LocalTransform, Height, Color). The
// worker world carries only the engine subset; the worker destructures ONLY the
// physics-bridge components (RigidBodyRef, RigidBodyState), which are present on both,
// so the wider static type is sound at every real call site.
export function getEngineComponents(world: World): EngineComponents {
  const components = (world as EngineWorldLike).components;
  if (!components) {
    throw new Error("Engine components are not available on this world");
  }
  return components as EngineComponents;
}

export function getEngineSab(world: World): EngineSab {
  const sab = (world as EngineWorldLike).sab;
  if (!sab) {
    throw new Error("Engine SAB is not available on this world");
  }
  return sab;
}
