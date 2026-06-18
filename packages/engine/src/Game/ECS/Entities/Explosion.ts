import { addEntity } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { VFXTypeValue } from "../Components/VFX.ts";
import { getGameComponents } from "../createGameWorld.ts";
import {
  addTransformComponents,
  applyMatrixTranslate,
  applyMatrixScale,
  LocalTransform,
} from "renderer/src/ECS/Components/Transform.ts";
import { ZIndex } from "../../consts.ts";
import { RenderDI } from "../../DI/RenderDI.ts";

export interface ExplosionOptions {
  x: number;
  y: number;
  type: VFXTypeValue;
  size: number;
  duration: number;
}

export function spawnExplosion(
  options: ExplosionOptions,
  { world } = GameDI,
  { enabled } = RenderDI,
) {
  if (!enabled) return;

  const { VFX, Progress, DestroyByTimeout } = getGameComponents(world);
  const eid = addEntity(world);

  addTransformComponents(world, eid);
  applyMatrixTranslate(LocalTransform.matrix.getBatch(eid), options.x, options.y, ZIndex.Explosion);
  applyMatrixScale(LocalTransform.matrix.getBatch(eid), options.size, options.size);

  VFX.addComponent(world, eid, options.type);
  Progress.addComponent(world, eid, options.duration);
  DestroyByTimeout.addComponent(world, eid, options.duration);
}
