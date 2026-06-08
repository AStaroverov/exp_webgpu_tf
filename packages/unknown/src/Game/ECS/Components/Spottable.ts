import { addComponent, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { NestedArray, TypedArray } from '../../../../../renderer/src/utils.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';
import { SpottingConfig } from '../../Config/gameplay.ts';

/** Max distinct spotters credited per victim (bounded by the opposing team's size). */
const MAX_SPOTTERS = 8;
const SPOTTER_ENTRY = 2; // (spotterPlayerId, accumulatedCredit)

/**
 * "How I am seen by the opposing side" — the spotted entity's own visibility facet.
 * The game is always two-sided, and a unit is only ever spotted by the OTHER side, so
 * visibility is a single per-victim value (no observing-team dimension). Holds two
 * sub-states, both keyed on this (victim) entity:
 *
 * 1. `confidence[eid]` — a single fading weight, GRADED by the source it was last
 *    reinforced by (`SpottingConfig.confidence`: beam 1, fire 0.5, proximity 0.25) and
 *    decaying to 0 over `SpottingConfig.memoryMs`. Each tick the spotting system decays
 *    it then reinforces it to the MAX currently-active source via `markSpotted(level)`,
 *    so the value floors at the strongest live source and fades from there once it is
 *    lost. This drives visibility: `isVisible` (= `confidence > 0`) gates the discrete
 *    Enemy/Hp/stat planes, and the value itself scales the heat / SpotConfidence channel.
 *
 * 2. spotter LEDGER — per CONTRIBUTING SPOTTER (playerId), a MONOTONIC accumulation of
 *    the per-tick confidence GAINS that spotter caused (proximity or searchlight). This
 *    is the single source of truth for spot-reward attribution: the spotting system,
 *    which already decides who contributes each tick, credits each contributor here, so
 *    the reward layer just diffs the ledger (mirrors `LastHitters`). The stored credit
 *    is NEUTRAL (raw confidence gain) — role rates (Ranger vs fighter) live in the
 *    training layer, not here, so the game stays reward-agnostic. A fire self-reveal
 *    (`revealByFire`) raises `confidence` with no contributor, so it touches no ledger
 *    entry and pays nothing.
 */
export const createSpottableComponent = defineComponent((Spottable) => {
    const confidence = TypedArray.f32(delegate.defaultSize);
    const spotters = NestedArray.f64(MAX_SPOTTERS * SPOTTER_ENTRY, delegate.defaultSize);

    return {
        confidence,
        spotters,

        addComponent(world: World, eid: number) {
            addComponent(world, eid, Spottable);
            confidence[eid] = 0;
            spotters.getBatch(eid).fill(0);
        },

        /**
         * Reinforce confidence in `eid` to at least `level`, the source strength
         * (`SpottingConfig.confidence.{beam,fire,proximity}`). Takes the max so a
         * stronger live source (or a not-yet-faded stronger memory) is never lowered
         * by a weaker one the same tick.
         */
        markSpotted(eid: number, level: number) {
            if (level > confidence[eid]) confidence[eid] = level;
        },

        decay(eid: number, delta: number) {
            const next = confidence[eid] - delta / SpottingConfig.memoryMs;
            confidence[eid] = next > 0 ? next : 0;
        },

        /** Whether the opposing side currently sees `eid` at all. */
        isVisible: (eid: number) => confidence[eid] > 0,
        /** Fading confidence (0..1) the opposing side has on `eid` — both "how I see an enemy" and "how the enemy sees me". */
        getConfidence: (eid: number) => confidence[eid],

        /**
         * Credit `playerId` with `amount` of spot contribution against victim `eid`
         * (monotonic — the reward layer reads the per-tick increase). Same fan-in
         * eviction as `LastHitters`: reuse the player's slot, else an empty one, else
         * overwrite the smallest-credit slot (does not fire at team-sized fan-in).
         */
        addSpotCredit(eid: number, playerId: number, amount: number) {
            const arr = spotters.getBatch(eid);

            for (let i = 0; i < MAX_SPOTTERS * SPOTTER_ENTRY; i += SPOTTER_ENTRY) {
                if (arr[i] === playerId) {
                    arr[i + 1] += amount;
                    return;
                }
            }
            for (let i = 0; i < MAX_SPOTTERS * SPOTTER_ENTRY; i += SPOTTER_ENTRY) {
                if (arr[i] === 0) {
                    arr[i] = playerId;
                    arr[i + 1] = amount;
                    return;
                }
            }
            let minIndex = 0;
            let minCredit = arr[1];
            for (let i = SPOTTER_ENTRY; i < MAX_SPOTTERS * SPOTTER_ENTRY; i += SPOTTER_ENTRY) {
                if (arr[i + 1] < minCredit) {
                    minCredit = arr[i + 1];
                    minIndex = i;
                }
            }
            arr[minIndex] = playerId;
            arr[minIndex + 1] = amount;
        },

        forEachSpotters(eid: number, callback: (playerId: number, credit: number) => void) {
            const arr = spotters.getBatch(eid);
            for (let i = 0; i < MAX_SPOTTERS * SPOTTER_ENTRY; i += SPOTTER_ENTRY) {
                if (arr[i] === 0) break;
                callback(arr[i], arr[i + 1]);
            }
        },
    };
});
