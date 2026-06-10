import { addEntity } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { getGameComponents } from "../createGameWorld.ts";
import {
  addTransformComponents,
  applyMatrixTranslate,
  LocalTransform,
} from "../../../../../renderer/src/ECS/Components/Transform.ts";
import { ShapeKind } from "../../../../../renderer/src/ECS/Components/Shape.ts";
import { RenderDI } from "../../DI/RenderDI.ts";

export interface LightFlashOptions {
  x: number;
  y: number;
  radius: number;
  duration: number;
  color: readonly [number, number, number];
  intensity: number;
}

// Emit-only companion for VFX flashes: an invisible SDF circle (alpha 0 -> discarded
// in the main pass) that feeds the radiance cascades emission pass. z = 0 keeps it
// out of the shadow-map passes.
export function spawnLightFlash(
  options: LightFlashOptions,
  { world } = GameDI,
  { enabled } = RenderDI,
) {
  if (!enabled) return;

  const { Shape, Color, LightEmitter, LightEmitterAnimation, Progress, DestroyByTimeout } =
    getGameComponents(world);
  const eid = addEntity(world);

  addTransformComponents(world, eid);
  applyMatrixTranslate(LocalTransform.matrix.getBatch(eid), options.x, options.y, 0);

  Shape.addComponent(world, eid, ShapeKind.Circle, options.radius);
  Color.addComponent(world, eid, options.color[0], options.color[1], options.color[2], 0);
  LightEmitter.addComponent(world, eid, options.intensity);
  // Animation track + Progress clock → createLightEmitterAnimationSystem decays
  // the light to zero over `duration` instead of a hard cutoff at destroy time.
  LightEmitterAnimation.addComponent(world, eid, options.intensity);
  Progress.addComponent(world, eid, options.duration);
  DestroyByTimeout.addComponent(world, eid, options.duration);
}
