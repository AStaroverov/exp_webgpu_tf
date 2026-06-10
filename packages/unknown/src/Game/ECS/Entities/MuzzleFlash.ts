import { addEntity } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { VFXType } from "../Components/VFX.ts";
import { getGameComponents } from "../createGameWorld.ts";
import {
  addTransformComponents,
  applyMatrixTranslate,
  applyMatrixRotateZ,
  applyMatrixScale,
  LocalTransform,
} from "../../../../../renderer/src/ECS/Components/Transform.ts";
import { ZIndex } from "../../consts.ts";
import { RenderDI } from "../../DI/RenderDI.ts";
import { FlashLightConfig } from "../../Config/index.ts";
import { spawnLightFlash } from "./LightFlash.ts";

export interface MuzzleFlashOptions {
  x: number;
  y: number;
  size: number;
  duration: number;
  rotation?: number;
}

export function spawnMuzzleFlash(
  options: MuzzleFlashOptions,
  { world } = GameDI,
  { enabled } = RenderDI,
) {
  if (!enabled) return;

  const { VFX, Progress, DestroyByTimeout } = getGameComponents(world);
  const eid = addEntity(world);

  addTransformComponents(world, eid);
  applyMatrixTranslate(
    LocalTransform.matrix.getBatch(eid),
    options.x,
    options.y,
    ZIndex.MuzzleFlash,
  );
  if (options.rotation) {
    applyMatrixRotateZ(LocalTransform.matrix.getBatch(eid), options.rotation);
  }
  applyMatrixScale(LocalTransform.matrix.getBatch(eid), options.size, options.size);

  VFX.addComponent(world, eid, VFXType.MuzzleFlash);
  Progress.addComponent(world, eid, options.duration);
  DestroyByTimeout.addComponent(world, eid, options.duration);

  spawnLightFlash({
    x: options.x,
    y: options.y,
    radius: options.size,
    duration: FlashLightConfig.muzzle.duration,
    color: FlashLightConfig.muzzle.color,
    intensity: FlashLightConfig.muzzle.intensity,
  });
}
