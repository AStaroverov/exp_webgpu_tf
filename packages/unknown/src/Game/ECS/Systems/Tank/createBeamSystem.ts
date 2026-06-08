/**
 * Beam pulse system — the Ranger beam behaves like a shot, not a permanent
 * floodlight: off by default, a Fire action pulses it on for `SpotlightConfig.pulseMs`.
 *
 * This system owns NO methods — it only drives entities each tick. The "trigger" and
 * "end" operations live as beam-entity methods (`activateBeam` / `deactivateBeam` in
 * `Beam.ts`); the pulse clock is the generic `Progress` component (advanced by
 * `createProgressSystem`).
 *
 * Runs each tick BEFORE the spotting system: the tick a pulse ends, it hides the
 * visual beam. The spotting system gates its per-Ranger beam query on `isBeamActive`,
 * so the beam only spots while lit.
 */

import { query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { getGameComponents } from '../../createGameWorld.ts';
import { deactivateBeam, isBeamActive } from '../../Entities/Beam.ts';

export function createBeamSystem({ world } = GameDI) {
    const { BeamRef } = getGameComponents(world);

    return function updateBeam() {
        const rangers = query(world, [BeamRef]);
        for (const eid of rangers) {
            // Idempotent: while lit the beam is already shown; once the pulse runs out
            // (or before the first shot) this hides it.
            if (!isBeamActive(eid)) deactivateBeam(eid);
        }
    };
}
