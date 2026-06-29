import { createWorld as createBitecsWorld, createEntityIndex, World } from "bitecs";
import { createRenderComponents, type RenderComponents } from "../../../renderer/src/ECS/world.ts";
import { createEngineComponents } from "./components.ts";
import { bindBundle, type Sab } from "../../../common/src/sab/sab.ts";
import { getComponentSab, setComponentSabFactory } from "../../../common/src/component.ts";
import { allocate, type SabBundle } from "../../../common/src/sab/registry.ts";
import { adoptEntity } from "../../../common/src/sab/adoptEntity.ts";

export type PhysicsComponents = ReturnType<typeof createEngineComponents>;
export type PhysicsWorkerWorld = World<{ components: PhysicsComponents }>;

export function createPhysicsWorkerWorld(bundle: SabBundle): PhysicsWorkerWorld {
  const context = { components: null as unknown as PhysicsComponents };
  const world = createBitecsWorld(createEntityIndex(), context) as PhysicsWorkerWorld;
  setComponentSabFactory(world, () => bindBundle(bundle, false));
  context.components = createEngineComponents(world);
  return world;
}

export type EngineComponents = RenderComponents & PhysicsComponents;
export type EngineWorld = World<{ components: EngineComponents }>;

export function createEngineWorld(): EngineWorld {
  const context = { components: null as unknown as EngineComponents };
  const world = createBitecsWorld(createEntityIndex(), context) as EngineWorld;
  setComponentSabFactory(world, () => bindBundle(allocate(), true));
  context.components = {
    ...createRenderComponents(world),
    ...createEngineComponents(world),
  };
  return world;
}

type EngineWorldLike = World<{ components: EngineComponents | PhysicsComponents }>;

export function getEngineComponents(world: World): EngineComponents {
  const components = (world as EngineWorldLike).components;
  if (!components) {
    throw new Error("Engine components are not available on this world");
  }
  return components as EngineComponents;
}

export function getEngineSab(world: World): Sab {
  const sab = getComponentSab(world);
  if (!sab) {
    throw new Error("Engine SAB is not available on this world");
  }
  return sab as Sab;
}

// Allocate the next shared-counter eid AND materialize it in this world. The SAB
// counter is the single eid authority across the main + worker worlds; callers just
// get a live eid back and never touch the SAB.
export function createEntityId(world: World): number {
  const eid = getEngineSab(world).nextEid();
  adoptEntity(world, eid);
  return eid;
}
