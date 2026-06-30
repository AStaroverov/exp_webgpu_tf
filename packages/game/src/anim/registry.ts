import { mat4, vec3 } from "gl-matrix";
import { BehaviorSubject } from "rxjs";
import {
  getEngineComponents,
  type EngineWorld,
} from "../../../engine/src/ECS/createEngineWorld.js";
import type { EntityInstance } from "../Entities/registry.js";
import { canonicalizeTrack, makeClipPlayer, type Clip, type ClipPlayer } from "./clip.js";

export type RegisteredClip = {
  name: string;
  loop: boolean;
  clip: Clip;
};

// In-session only (NOT persisted): entity type id -> clip name -> clip.
const clipsByEntity = new Map<string, Map<string, RegisteredClip>>();
export const clips$ = new BehaviorSubject<void>(undefined);

export function registerClip(entityId: string, registered: RegisteredClip): void {
  for (const track of registered.clip.tracks) canonicalizeTrack(track);
  let byName = clipsByEntity.get(entityId);
  if (byName === undefined) {
    byName = new Map();
    clipsByEntity.set(entityId, byName);
  }
  byName.set(registered.name, registered);
  clips$.next();
}

export function getClips(entityId: string): RegisteredClip[] {
  const byName = clipsByEntity.get(entityId);
  return byName === undefined ? [] : [...byName.values()];
}

// Resolve a clip's tracks against a freshly built instance: trackEid by bone name, restScale
// snapshotted from each bone's live matrix BEFORE any clip has run, then build the player.
export function buildClipPlayer(
  world: EngineWorld,
  clip: Clip,
  instance: Pick<EntityInstance, "root" | "bones">,
): ClipPlayer {
  const { LocalTransform } = getEngineComponents(world);
  const trackEid = clip.tracks.map((t) => instance.bones[t.bone] ?? -1);
  const restScale = trackEid.map((eid) =>
    eid < 0
      ? vec3.fromValues(1, 1, 1)
      : mat4.getScaling(vec3.create(), LocalTransform.matrix.getBatch(eid)),
  );
  return makeClipPlayer(world, clip, trackEid, restScale);
}

// Turn each registered clip into a (delta) => void that advances its own clip clock and
// calls the player, so it plugs into the existing (delta)-driven animations Record and
// coexists with the hand-coded procedural closures.
export function makeClipAnimations(
  world: EngineWorld,
  entityId: string,
  instance: Pick<EntityInstance, "root" | "bones">,
): Record<string, (delta: number) => void> {
  const animations: Record<string, (delta: number) => void> = {};

  for (const registered of getClips(entityId)) {
    const { clip, loop } = registered;
    const player = buildClipPlayer(world, clip, instance);

    let clock = 0;
    animations[registered.name] = (delta) => {
      clock += delta;
      const phase = loop
        ? (clock % clip.duration) / clip.duration
        : Math.min(clock / clip.duration, 1);
      player(phase);
    };
  }

  return animations;
}
