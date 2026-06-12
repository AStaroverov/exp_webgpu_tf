import { query } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { getGameComponents } from "../../createGameWorld.ts";

/**
 * Render-only: animates every `LightEmitter` that carries a `LightEmitterAnimation`
 * track, clocked by `Progress` — light flashes (explosion / muzzle / hit) decay
 * smoothly instead of being cut off at full brightness by `DestroyByTimeout`.
 * Quadratic ease-out from the track's peak values: a bright pop that dims fast
 * at first, then tails off to zero right when the entity dies.
 * Runs after `updateProgress` so it sees this frame's age.
 */
export function createLightEmitterAnimationSystem({ world } = GameDI) {
  const { LightEmitter, LightEmitterAnimation, Progress } = getGameComponents(world);

  return () => {
    const eids = query(world, [LightEmitter, LightEmitterAnimation, Progress]);

    for (let i = 0; i < eids.length; i++) {
      const eid = eids[i];
      const left = 1 - Math.min(1, Progress.getProgress(eid));
      const k = left * left;
      // set$ (not a raw write): the SDF pass uploads intensity only onSet.
      LightEmitter.set$(
        eid,
        LightEmitterAnimation.intensity.get(eid) * k,
        LightEmitterAnimation.radius.get(eid) * k,
      );
    }
  };
}
