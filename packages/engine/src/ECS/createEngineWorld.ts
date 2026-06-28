import { createWorld as createBitecsWorld, createEntityIndex, World } from "bitecs";
import {
  createRenderComponents,
  type RenderComponents,
} from "../../../renderer3d_2/src/ECS/world.ts";
import { createEngineComponents, type EngineComponentsLocal } from "./components.ts";
import type { EngineSab, PhysicsRole } from "../sab/engineSab.ts";
import { getComponentSab } from "../../../renderer3d_2/src/ECS/utils.ts";
import type { SabBundle } from "../../../renderer3d_2/src/sab/registry.ts";

export type EngineComponents = RenderComponents & ReturnType<typeof createEngineComponents>;

export type EngineWorld = World<{
  components: EngineComponents;
  physicsRole: PhysicsRole;
  time: {
    delta: number;
    elapsed: number;
    last: number;
  };
}>;

export function createEngineWorld(): EngineWorld {
  const context = {
    components: null as unknown as EngineComponents,
    physicsRole: { kind: "producer" },
    time: {
      delta: 0,
      elapsed: 0,
      last: performance.now(),
    },
  };
  const world = createBitecsWorld(createEntityIndex(), context) as EngineWorld;
  context.components = {
    ...createRenderComponents(world),
    ...createEngineComponents(world),
  };
  return world;
}

export type PhysicsWorkerWorld = World<{
  components: EngineComponentsLocal;
  physicsRole: PhysicsRole;
  time: {
    delta: number;
    elapsed: number;
    last: number;
  };
}>;

export function createPhysicsWorkerWorld(bundle: SabBundle): PhysicsWorkerWorld {
  const context = {
    components: null as unknown as EngineComponentsLocal,
    physicsRole: { kind: "consumer", bundle },
    time: {
      delta: 0,
      elapsed: 0,
      last: 0,
    },
  };
  const world = createBitecsWorld(createEntityIndex(), context) as PhysicsWorkerWorld;
  context.components = createEngineComponents(world);
  return world;
}

type EngineWorldLike = World<{
  components?: EngineComponents | EngineComponentsLocal;
}>;

export function getEngineComponents(world: World): EngineComponents {
  const components = (world as EngineWorldLike).components;
  if (!components) {
    throw new Error("Engine components are not available on this world");
  }
  return components as EngineComponents;
}

export function getEngineSab(world: World): EngineSab {
  const sab = getComponentSab(world);
  if (!sab) {
    throw new Error("Engine SAB is not available on this world");
  }
  return sab as EngineSab;
}

export function createEntityId(world: World): number {
  return getEngineSab(world).nextEid();
}
