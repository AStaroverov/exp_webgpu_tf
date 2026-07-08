import type GUI from "lil-gui";
import { getRenderComponents, type RenderWorld } from "../../ECS/world.ts";
import { SunLight } from "../../ECS/Systems/SunLight.ts";
import { setMatrixTranslate } from "../../ECS/Components/Transform.ts";
import { createRectangle, createSphere } from "../../ECS/Entities/Shapes.ts";
import type { DemoScene } from "./types.ts";

// "emitter" — ground + box occluder + a GUI-movable/resizable emitter sphere. The single-light
// laboratory: isolate one emitter (sun off) and drag it around to read its light/shadow.
export function createEmitterScene(world: RenderWorld): DemoScene {
  const { LocalTransform, LightEmitter, Shape } = getRenderComponents(world);

  // Live-edited from the GUI: position via the transform (re-uploaded every frame), radius via
  // the Shape setter (onSet → re-collected), intensity via LightEmitter.set$.
  const emitterCfg = { x: -6, y: 0, z: 2.5, radius: 2.5, intensity: 2.0 };

  SunLight.enabled = false; // isolate the single emitter
  createRectangle(world, {
    x: 0,
    y: 0,
    z: -0.5,
    width: 40,
    height: 40,
    depth: 0.5,
    color: [0.5, 0.5, 0.5, 1],
  });
  createRectangle(world, {
    x: 4,
    y: 0,
    z: 0,
    width: 3,
    height: 3,
    depth: 4,
    color: [0.8, 0.8, 0.8, 1],
  });
  const emitterId = createSphere(world, {
    x: emitterCfg.x,
    y: emitterCfg.y,
    z: emitterCfg.z,
    radius: emitterCfg.radius,
    color: [1, 1, 1, 1],
  });
  LightEmitter.addComponent(world, emitterId, emitterCfg.intensity);

  return {
    // Live emitter controls. Position writes the transform (re-uploaded every frame); radius
    // drives the sphere SDF (Shape.setSphere$ — the sphere's radius is its own full Z extent);
    // intensity via LightEmitter.set$.
    setupGUI(gui: GUI) {
      const moveEmitter = () =>
        setMatrixTranslate(
          LocalTransform.matrix.getBatch(emitterId),
          emitterCfg.x,
          emitterCfg.y,
          emitterCfg.z,
        );
      const resizeEmitter = () => {
        Shape.setSphere$(emitterId, emitterCfg.radius);
      };
      const em = gui.addFolder("Emitter");
      em.add(emitterCfg, "x", -20, 20, 0.1).onChange(moveEmitter);
      em.add(emitterCfg, "y", -20, 20, 0.1).onChange(moveEmitter);
      em.add(emitterCfg, "z", 0, 12, 0.1).onChange(moveEmitter);
      em.add(emitterCfg, "radius", 0.2, 8, 0.1).onChange(resizeEmitter);
      em.add(emitterCfg, "intensity", 0, 8, 0.1).onChange(() =>
        LightEmitter.set$(emitterId, emitterCfg.intensity, emitterCfg.radius),
      );
    },
  };
}
