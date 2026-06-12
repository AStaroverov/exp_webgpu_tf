import { hasComponent, query } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { getGameComponents } from "../createGameWorld.ts";
import { max } from "../../../../../../lib/math.ts";

/**
 * Drains each sensor projectile's `SensorHits` ring (filled by `hit$` during
 * `physicalFrame`): a part overlap (any team — friendly fire is on, only the
 * firer's own parts are exempt) deals the projectile's `Damagable` as an
 * instant hit (final damage + kind into the hitable pipeline, which owns the
 * kind specialties) and stamps its `Dotable` as a `Dot` on the part; an
 * obstacle overlap kills the projectile. No damage-kind logic lives here.
 */
export function createApplySensorHitsSystem({ world } = GameDI) {
  const { SensorHits, Damagable, Dotable, Dot, VehiclePart, Obstacle, DestroyByTimeout, Hitable } =
    getGameComponents(world);

  return () => {
    const projectileEids = query(world, [SensorHits]);

    for (let i = 0; i < projectileEids.length; i++) {
      const projectileEid = projectileEids[i];
      const hitCount = SensorHits.hitIndex[projectileEid];
      if (hitCount === 0) continue;

      for (let j = 0; j < hitCount; j++) {
        const otherEid = SensorHits.hits.get(projectileEid, j);

        if (hasComponent(world, otherEid, VehiclePart)) {
          // Instant damage FROM the projectile: final value + kind; the source
          // carries the firer's PlayerRef/TeamRef, so LastHitters credits the
          // attacker even at 0 damage.
          if (hasComponent(world, otherEid, Hitable)) {
            Hitable.hit$(
              otherEid,
              projectileEid,
              Damagable.damage[projectileEid],
              Damagable.kind[projectileEid],
            );
          }

          // Stamp the damage-over-time: refresh duration, keep the strongest dps
          // (a part without a Dot reads 0).
          if (hasComponent(world, projectileEid, Dotable)) {
            const dps = max(Dot.dps.get(otherEid), Dotable.dps[projectileEid]);
            Dot.refresh(
              otherEid,
              Dotable.kind[projectileEid],
              dps,
              Dotable.durationMs[projectileEid],
            );
          }

          // Pass-through decay: each enemy-part overlap eats the projectile's lifetime.
          DestroyByTimeout.resetTimeout(
            projectileEid,
            max(
              0,
              DestroyByTimeout.timeout[projectileEid] - SensorHits.hitLifeCostMs[projectileEid],
            ),
          );
        } else if (hasComponent(world, otherEid, Obstacle)) {
          DestroyByTimeout.resetTimeout(projectileEid, 0); // dies at the wall, no stamp
        }
      }

      SensorHits.resetHits(projectileEid);
    }
  };
}
