import { addEntity } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { VFXType } from "../Components/VFX.ts";
import { getGameComponents } from "../createGameWorld.ts";
import {
  addTransformComponents,
  applyMatrixTranslate,
  applyMatrixScale,
  LocalTransform,
} from "../../../../../renderer/src/ECS/Components/Transform.ts";
import { ZIndex } from "../../consts.ts";
import { RenderDI } from "../../DI/RenderDI.ts";
import { FlashLightConfig } from "../../Config/index.ts";
import { spawnLightFlash } from "./LightFlash.ts";

export interface HitFlashOptions {
  x: number;
  y: number;
  size: number;
  duration: number;
}

export function spawnHitFlash(
  options: HitFlashOptions,
  { world } = GameDI,
  { enabled } = RenderDI,
) {
  if (!enabled) return;

  const { VFX, Progress, DestroyByTimeout } = getGameComponents(world);
  const eid = addEntity(world);

  addTransformComponents(world, eid);
  applyMatrixTranslate(LocalTransform.matrix.getBatch(eid), options.x, options.y, ZIndex.HitFlash);
  applyMatrixScale(LocalTransform.matrix.getBatch(eid), options.size, options.size);

  VFX.addComponent(world, eid, VFXType.HitFlash);
  Progress.addComponent(world, eid, options.duration);
  DestroyByTimeout.addComponent(world, eid, options.duration);

  spawnLightFlash({
    x: options.x,
    y: options.y,
    radius: options.size,
    duration: FlashLightConfig.hit.duration,
    color: FlashLightConfig.hit.color,
    intensity: FlashLightConfig.hit.intensity,
  });
}
