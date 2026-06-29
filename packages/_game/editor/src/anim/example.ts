import { registerClip, type RegisteredClip } from "./registry.ts";

// Demo clip proving data-driven playback end to end: a 2-key right-arm raise,
// rest (arm down, identity rotation) -> 90deg pitch about X and back, looping.
// Targets only the arm bone, so it obeys the one-writer-per-bone rule (the unit/swordsman
// root stays owned by the procedural closures).
function armRaise(bone: string): RegisteredClip {
  return {
    name: "armRaise*",
    loop: true,
    clip: {
      duration: 2,
      tracks: [
        {
          bone,
          keys: [
            { t: 0, pos: [0.9, 0, 0.99], rot: [0, 0, 0, 1] },
            { t: 0.5, pos: [0.9, 0, 0.99], rot: [Math.SQRT1_2, 0, 0, Math.SQRT1_2] },
            { t: 1, pos: [0.9, 0, 0.99], rot: [0, 0, 0, 1] },
          ],
        },
      ],
    },
  };
}

export function registerDemoClips(): void {
  registerClip("unit", armRaise("armR"));
  registerClip("swordsman", armRaise("unit/armR"));
}
