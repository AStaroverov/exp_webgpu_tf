import { query, removeComponent } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { getGameComponents } from '../createGameWorld.ts';

/**
 * Records this tick's damage for every part under a `Dot` and expires it.
 * Damage goes through `Hitable.hit$` like every other damage type — the hitable
 * pass applies it and triggers the kind's specialty (accepting a harmless
 * one-frame lag). The self-`eid` source attributes nothing in `LastHitters`
 * (same team); attribution happened at stamp time from the projectile.
 */
export function createDotSystem({ world } = GameDI) {
    const { Dot, Hitable } = getGameComponents(world);

    return (delta: number) => {
        const eids = query(world, [Dot, Hitable]);

        for (let i = eids.length - 1; i >= 0; i--) {
            const eid = eids[i];

            Hitable.hit$(eid, eid, Dot.dps[eid] * delta / 1000, Dot.kind[eid]);

            Dot.remaining[eid] -= delta;
            if (Dot.remaining[eid] <= 0) {
                removeComponent(world, eid, Dot);
            }
        }
    };
}
