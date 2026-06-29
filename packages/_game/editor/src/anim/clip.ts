import { mat4, quat, vec3 } from "gl-matrix";
import {
  getEngineComponents,
  type EngineWorld,
} from "../../../../engine/src/ECS/createEngineWorld.ts";

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number]; // gl-matrix order [x, y, z, w]

export type Keyframe = {
  t: number; // normalized phase [0,1] within the clip — absolute seconds = t * clip.duration
  pos: Vec3;
  rot: Quat;
  scale?: number;
};

export type Track = {
  bone: string;
  keys: Keyframe[]; // sorted ascending by t
};

export type Clip = {
  duration: number; // seconds for one cycle; the keyframe phases are duration-independent
  tracks: Track[];
};

// Make neighbouring keys share a hemisphere so the quaternion spline stays on the short arc
// (and slerp would too). Run once at capture/load.
export function canonicalizeTrack(track: Track): void {
  const keys = track.keys;
  for (let i = 1; i < keys.length; i++) {
    if (quat.dot(keys[i - 1].rot, keys[i].rot) < 0) {
      const r = keys[i].rot;
      r[0] = -r[0];
      r[1] = -r[1];
      r[2] = -r[2];
      r[3] = -r[3];
    }
  }
}

function catmull(p0: number, p1: number, p2: number, p3: number, u: number): number {
  const u2 = u * u;
  const u3 = u2 * u;
  return (
    0.5 *
    (2 * p1 +
      (-p0 + p2) * u +
      (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 +
      (-p0 + 3 * p1 - 3 * p2 + p3) * u3)
  );
}

// phase ∈ [0,1] (clip-relative). Times live as normalized phases so changing clip.duration
// stretches/squashes the whole clip without touching the keys.
//
// Interpolation is Catmull-Rom: the curve PASSES THROUGH every keyframe but keeps a continuous
// velocity at the joints (C1), so the motion flows as one stroke. Per-segment ease-in-out would
// instead zero the velocity at every key — the part stops at each pose ("walking the points").
// Quaternions are splined component-wise (safe: canonicalizeTrack put neighbours in one hemisphere)
// then renormalized. Endpoints clamp their missing neighbour (p0=p1 at the start, p3=p2 at the end).
export function sampleTrack(track: Track, phase: number, outPos: vec3, outRot: quat): void {
  const keys = track.keys;
  const n = keys.length;
  if (n === 1) {
    vec3.set(outPos, keys[0].pos[0], keys[0].pos[1], keys[0].pos[2]);
    quat.set(outRot, keys[0].rot[0], keys[0].rot[1], keys[0].rot[2], keys[0].rot[3]);
    return;
  }

  let i = 0;
  while (i < n - 2 && phase > keys[i + 1].t) i++;
  const a = keys[i];
  const b = keys[i + 1];
  const k0 = keys[i > 0 ? i - 1 : i];
  const k3 = keys[i + 2 < n ? i + 2 : i + 1];

  const span = b.t - a.t;
  const raw = span > 0 ? (phase - a.t) / span : 0;
  const u = raw < 0 ? 0 : raw > 1 ? 1 : raw;

  for (let k = 0; k < 3; k++) outPos[k] = catmull(k0.pos[k], a.pos[k], b.pos[k], k3.pos[k], u);
  for (let k = 0; k < 4; k++) outRot[k] = catmull(k0.rot[k], a.rot[k], b.rot[k], k3.rot[k], u);
  quat.normalize(outRot, outRot);
}

// weight ∈ [0,1] blends the clip pose ONTO whatever is already in the bone's matrix (lerp pos /
// slerp rot): 1 = full clip, 0 = leave the existing pose untouched. Used to ease a clip in/out over
// a rest (or another stance) pose — the caller runs the player every frame with an eased weight.
export type ClipPlayer = (phase: number, weight?: number) => void;

export function makeClipPlayer(
  world: EngineWorld,
  clip: Clip,
  trackEid: number[],
  restScale: vec3[],
): ClipPlayer {
  const { LocalTransform } = getEngineComponents(world);
  const pos = vec3.create();
  const rot = quat.create();
  const curPos = vec3.create();
  const curRot = quat.create();

  return (phase: number, weight = 1) => {
    for (let i = 0; i < clip.tracks.length; i++) {
      const eid = trackEid[i];
      if (eid < 0) continue;
      const track = clip.tracks[i];
      if (track.keys.length === 0) continue;
      const m = LocalTransform.matrix.getBatch(eid);
      sampleTrack(track, phase, pos, rot);
      if (weight < 1) {
        mat4.getTranslation(curPos, m);
        mat4.getRotation(curRot, m);
        vec3.lerp(pos, curPos, pos, weight);
        quat.slerp(rot, curRot, rot, weight);
      }
      mat4.fromRotationTranslationScale(m, rot, pos, restScale[i]);
    }
  };
}
