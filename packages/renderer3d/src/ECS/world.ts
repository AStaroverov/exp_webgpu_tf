import { createWorld as createBitecsWorld, World } from "bitecs";
import {
  createBlurnessComponent,
  createColorComponent,
  createHeightComponent,
  createLightEmitterComponent,
  createRoundnessComponent,
  createThinnessComponent,
  createTranslucencyComponent,
} from "./Components/Common.ts";
import { createRopeComponent } from "./Components/Rope.ts";
import { createShapeComponent } from "./Components/Shape.ts";
import { GlobalTransform, LocalTransform } from "./Components/Transform.ts";

export function createRenderComponents(world: World) {
  return {
    GlobalTransform,
    LocalTransform,
    Rope: createRopeComponent(world),
    Shape: createShapeComponent(world),
    Color: createColorComponent(world),
    Thinness: createThinnessComponent(world),
    Roundness: createRoundnessComponent(world),
    Blurness: createBlurnessComponent(world),
    Height: createHeightComponent(world),
    LightEmitter: createLightEmitterComponent(world),
    Translucency: createTranslucencyComponent(world),
  };
}

export type RenderComponents = ReturnType<typeof createRenderComponents>;

export type RenderWorldLike = World<{
  components?: RenderComponents;
}>;

export type RenderWorld = World<{
  components: RenderComponents;
  time: {
    delta: number;
    elapsed: number;
    then: number;
  };
}>;

export function createWorld(): RenderWorld {
  const context = {
    components: null as unknown as RenderComponents,
    time: {
      delta: 0,
      elapsed: 0,
      then: performance.now(),
    },
  };
  context.components = createRenderComponents(context as RenderWorld);

  return createBitecsWorld(context);
}

export function getRenderComponents(world: World): RenderComponents {
  const components = (world as RenderWorldLike).components;
  if (!components) {
    throw new Error("Renderer components are not available on this world");
  }
  return components;
}

export type { World };
