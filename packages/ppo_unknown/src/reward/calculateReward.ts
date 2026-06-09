/**
 * Reward shaping for ppo_unknown — ports the tanks `calculateReward` concept:
 * the step reward is the DELTA of a cumulative, event-based combat score
 * (`ScoreTracker`), not an hp count. Only the simplest events are scored (hits,
 * kills, approach-to-nearest-enemy) — all physics/realtime shaping from tanks
 * (aim alignment, turret tracking, proximity) is intentionally removed.
 *
 *   calculateActionReward(eid) — cumulative score for the tank's player (tanks'
 *                                `Score.getTotalScore`); the agent subtracts the
 *                                previous value to get the per-macro-action reward.
 *   getFramePenalty(frame)     — tiny per-decision time cost.
 *   calculateFinalReward(...)  — episode outcome: a continuous reward proportional
 *                                to the team's success ratio (relative surviving-health
 *                                advantage in [-1, 1]); a positive outcome is split by
 *                                combat contribution, a loss/draw is shared equally.
 */

import { clamp } from 'lodash';
import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';
import { getTankTeamId } from '../../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { scoreTracker } from './ScoreTracker.ts';
import type { UnknownAgent } from '../env/UnknownAgent.ts';

/**
 * Dense-shaping anneal: the per-macro-action reward (hit/kill/approach delta) is
 * faded with the network iteration so late training leans on the terminal win/loss
 * signal. Multiplicative, applied to the WHOLE shaping delta in `UnknownAgent`, so
 * `ScoreTracker`'s raw score (used here for win-contribution weighting) stays intact.
 *   weight = 1 until SHAPING_FULL_UNTIL, then linear to SHAPING_FLOOR at SHAPING_ZERO_AT.
 */
const SHAPING_FLOOR = 0.1;
const SHAPING_FULL_UNTIL = 50_000;
const SHAPING_ZERO_AT = 150_000;

/** Multiplicative weight on the dense shaping reward for the given network iteration. */
export function getShapingWeight(iteration: number): number {
    const t = clamp((iteration - SHAPING_FULL_UNTIL) / (SHAPING_ZERO_AT - SHAPING_FULL_UNTIL), 0, 1);
    return 1 + (SHAPING_FLOOR - 1) * t;
}

/** Reward magnitude at a perfect outcome (success ratio = ±1). */
const WIN_REWARD = 3;
/** Team-spirit τ: 0 = selfish, 1 = fully cooperative. Fixed for the MVP scenario. */
const TEAM_SPIRIT = 0.5;

/** Cumulative combat score for the tank's player (delta'd by the agent). */
export function calculateActionReward(eid: number, { world } = GameDI): number {
    const { PlayerRef } = getGameComponents(world);
    return scoreTracker.getScore(PlayerRef.id[eid]);
}

/**
 * Episode-end reward for `eid`. `successRatioTeam0` is the episode success ratio from
 * team 0's perspective in [-1, 1] (Scenario.getSuccessRatio): the relative
 * surviving-health advantage, +1 = team 0 intact while team 1 is wiped.
 *
 * The team reward is continuous: WIN_REWARD × success ratio from THIS tank's team
 * perspective (+1 = enemy wiped while we're intact, -1 = the reverse, 0 = even trade).
 * A positive outcome is then weighted by the tank's relative combat-score contribution
 * (team-spirit blended; team average stays the team reward). A loss/draw is shared
 * equally — the tank that fought hardest is not punished more than the one that hid.
 */
export function calculateFinalReward(
    eid: number,
    successRatioTeam0: number,
    agents: UnknownAgent[],
    { world } = GameDI,
): number {
    const myTeam = getTankTeamId(eid);
    const successRatio = myTeam === 0 ? successRatioTeam0 : -successRatioTeam0;
    const teamReward = WIN_REWARD * successRatio;
    if (teamReward <= 0) return teamReward;

    const { PlayerRef } = getGameComponents(world);
    const teammates = agents.filter((a) => getTankTeamId(a.tankEid) === myTeam);
    const teamSize = Math.max(1, teammates.length);

    const myScore = scoreTracker.getScore(PlayerRef.id[eid]);
    const totalTeamScore = teammates.reduce(
        (sum, a) => sum + scoreTracker.getScore(PlayerRef.id[a.tankEid]),
        0,
    );
    // Relative contribution: 1 = team average, teamSize = did everything alone.
    const relativeShare = totalTeamScore > 0
        ? (myScore / totalTeamScore) * teamSize
        : 1;
    const contribution = (1 - TEAM_SPIRIT) * relativeShare + TEAM_SPIRIT;

    return teamReward * contribution;
}
