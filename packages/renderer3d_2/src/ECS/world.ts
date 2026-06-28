import { createWorld as createBitecsWorld, World } from "bitecs";
import {
  createBlurnessComponent,
  createColorComponent,
  createLightEmitterComponent,
  createRoundnessComponent,
  createThinnessComponent,
  createTranslucencyComponent,
} from "./Components/Common.ts";
import { createRopeComponent } from "./Components/Rope.ts";
import { createShapeComponent } from "./Components/Shape.ts";
import {
  createGlobalTransformComponent,
  createLocalTransformComponent,
} from "./Components/Transform.ts";

export function createRenderComponents(world: World) {
  return {
    GlobalTransform: createGlobalTransformComponent(),
    LocalTransform: createLocalTransformComponent(),
    Rope: createRopeComponent(world),
    Shape: createShapeComponent(world),
    Color: createColorComponent(world),
    Thinness: createThinnessComponent(world),
    Roundness: createRoundnessComponent(world),
    Blurness: createBlurnessComponent(world),
    LightEmitter: createLightEmitterComponent(world),
    Translucency: createTranslucencyComponent(world),
  };
}

export type RenderComponents = ReturnType<typeof createRenderComponents>;

export type RenderWorld = World<{
  components: RenderComponents;
}>;

export function createWorld(): RenderWorld {
  const context = {
    components: null as unknown as RenderComponents,
  };
  context.components = createRenderComponents(context as RenderWorld);

  return createBitecsWorld(context);
}

export function getRenderComponents(world: World): RenderComponents {
  const components = (world as RenderWorld).components;
  if (!components) {
    throw new Error("Renderer components are not available on this world");
  }
  return components;
}

export type { World };
