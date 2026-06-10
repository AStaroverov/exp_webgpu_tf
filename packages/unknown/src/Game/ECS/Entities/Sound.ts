import { addEntity, EntityId, hasComponent } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { SoundType } from "../Components/Sound.ts";
import { getGameComponents } from "../createGameWorld.ts";
import {
  addTransformComponents,
  applyMatrixTranslate,
  LocalTransform,
} from "renderer/src/ECS/Components/Transform.ts";

export interface SpawnSoundOptions {
  type: SoundType;
  volume?: number;
  loop?: boolean;
  autoplay?: boolean;
  destroyOnFinish?: boolean;
}

export interface SpawnSoundAtPositionOptions extends SpawnSoundOptions {
  x: number;
  y: number;
}

export interface SpawnSoundWithParentOptions extends SpawnSoundOptions {
  parentEid: EntityId;
}

export function spawnSoundAtPosition(
  options: SpawnSoundAtPositionOptions,
  { world } = GameDI,
): EntityId {
  const { Sound, DestroyOnSoundFinish } = getGameComponents(world);
  const eid = addEntity(world);

  addTransformComponents(world, eid);
  applyMatrixTranslate(LocalTransform.matrix.getBatch(eid), options.x, options.y, 0);

  Sound.addComponent(world, eid, options.type, {
    loop: options.loop ?? false,
    volume: options.volume ?? 1,
    autoplay: options.autoplay ?? true,
  });

  if (options.destroyOnFinish) {
    DestroyOnSoundFinish.addComponent(world, eid);
  }

  return eid;
}

export function spawnSoundAtParent(
  options: SpawnSoundWithParentOptions,
  { world } = GameDI,
): EntityId {
  const { Sound, DestroyOnSoundFinish, Parent, Children } = getGameComponents(world);
  const eid = addEntity(world);

  if (!hasComponent(world, options.parentEid, Children)) {
    Children.addComponent(world, options.parentEid);
  }

  Parent.addComponent(world, eid, options.parentEid);
  Children.addChildren(options.parentEid, eid);

  Sound.addComponent(world, eid, options.type, {
    loop: options.loop ?? true,
    volume: options.volume ?? 1,
    autoplay: options.autoplay ?? true,
  });

  if (options.destroyOnFinish) {
    DestroyOnSoundFinish.addComponent(world, eid);
  }

  return eid;
}
