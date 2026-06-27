import { createWorld as createBitecsWorld, World } from "bitecs";
import {
  createRenderComponents,
  type RenderComponents,
} from "../../../renderer3d_2/src/ECS/world.ts";
import { createEngineComponents } from "./components.ts";

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
  time: {
    delta: number;
    elapsed: number;
    last: number;
  };
}>;

export function createEngineWorld(): EngineWorld {
  const context = {
    components: null as unknown as EngineComponents,
    time: {
      delta: 0,
      elapsed: 0,
      last: performance.now(),
    },
  };
  const world = createBitecsWorld(context) as EngineWorld;
  context.components = {
    ...createRenderComponents(world),
    ...createEngineComponents(world),
  };
  return world;
}

type EngineWorldLike = World<{ components?: EngineComponents }>;

export function getEngineComponents(world: World): EngineComponents {
  const components = (world as EngineWorldLike).components;
  if (!components) {
    throw new Error("Engine components are not available on this world");
  }
  return components;
}
