import { mat4, quat, vec3 } from "gl-matrix";
import {
  getEngineComponents,
  type EngineWorld,
} from "../../../../engine/src/ECS/createEngineWorld.ts";
import type { EntityInstance } from "../Entities/registry.ts";
import type { Clip, Quat, Vec3 } from "./clip.ts";

// Authoring model: each Snapshot appends a RECORD — a whole-entity pose (every animatable bone's
// transform) tagged with a keyframe `key` and a `pct` of the total duration. Records that share a
// `key` collapse into one keyframe, merging bone-wise in record order (later wins), so a keyframe
// can be built up over several snapshots. This is what the right-panel pipeline edits; it converts
// to the track-based Clip the player + log consume.
export type SnapRecord = { key: number; pct: number; pose: Record<string, { pos: Vec3; rot: Quat }> };

export type EditClip = {
  entityId: string;
  name: string;
  duration: number;
  bones: string[];
  records: SnapRecord[];
};

export function animatableBones(instance: EntityInstance): string[] {
  return Object.keys(instance.bones).filter((b) => b !== "root" && !b.endsWith("/root"));
}

const sp = vec3.create();
const sr = quat.create();

export function snapshotPose(
  world: EngineWorld,
  instance: EntityInstance,
  bones: string[],
): Record<string, { pos: Vec3; rot: Quat }> {
  const { LocalTransform } = getEngineComponents(world);
  const pose: Record<string, { pos: Vec3; rot: Quat }> = {};
  for (const bone of bones) {
    const m = LocalTransform.matrix.getBatch(instance.bones[bone]);
    mat4.getTranslation(sp, m);
    mat4.getRotation(sr, m);
    pose[bone] = { pos: [sp[0], sp[1], sp[2]], rot: [sr[0], sr[1], sr[2], sr[3]] };
  }
  return pose;
}

export function editToClip(edit: EditClip): Clip {
  const groups = new Map<number, { pct: number; pose: Record<string, { pos: Vec3; rot: Quat }> }>();
  for (const rec of edit.records) {
    const group = groups.get(rec.key) ?? { pct: rec.pct, pose: {} };
    group.pct = rec.pct;
    for (const bone of edit.bones) {
      if (rec.pose[bone] !== undefined) group.pose[bone] = rec.pose[bone];
    }
    groups.set(rec.key, group);
  }
  const keyframes = [...groups.values()];
  const tracks = edit.bones.map((bone) => ({
    bone,
    keys: keyframes
      .filter((g) => g.pose[bone] !== undefined)
      .map((g) => ({
        t: Math.min(1, Math.max(0, g.pct / 100)),
        pos: g.pose[bone].pos,
        rot: g.pose[bone].rot,
      }))
      .sort((a, b) => a.t - b.t),
  }));
  return { duration: edit.duration, tracks };
}
