import { canonicalizeTrack, type Clip, type Quat, type Track, type Vec3 } from "../clip.ts";
import swordSwing from "./swordmans_sword_swing.json";

type RawClip = {
  loop?: boolean;
  duration: number;
  tracks: { bone: string; keys: { t: number; pos: number[]; rot: number[] }[] }[];
};

// The authored clips are one-way (cocked → strike, keys spanning the full [0,1]); to loop them we
// squeeze the forward motion into [0, 1/(1+RETURN_RATIO)] and append a closing key (= each track's
// first key) at t=1. The player then eases the part back to its start pose over the tail and the
// cycle seam is identity (no snap). `duration` grows by the same ratio so the forward speed is kept.
const RETURN_RATIO = 0.5;

function loopBack(raw: RawClip, returnRatio: number): Clip {
  const forwardFrac = 1 / (1 + returnRatio);
  const tracks: Track[] = raw.tracks.map((t) => ({
    bone: t.bone,
    keys: t.keys.map((k) => ({
      t: k.t * forwardFrac,
      pos: [k.pos[0], k.pos[1], k.pos[2]] as Vec3,
      rot: [k.rot[0], k.rot[1], k.rot[2], k.rot[3]] as Quat,
    })),
  }));
  for (const track of tracks) {
    const first = track.keys[0];
    track.keys.push({ t: 1, pos: [...first.pos], rot: [...first.rot] });
    canonicalizeTrack(track);
  }
  return { duration: raw.duration * (1 + returnRatio), tracks };
}

export const SWORD_SWING: Clip = loopBack(swordSwing, RETURN_RATIO);
