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
 *   calculateFinalReward(...)  — episode outcome: win (enemy team ≥80% destroyed)
 *                                pays a contribution-weighted reward; anything short
 *                                of a win costs a penalty that shrinks linearly with
 *                                progress towards the win threshold.
 */

import { clamp } from 'lodash';
import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';
import { getTankTeamId } from '../../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { scoreTracker } from './ScoreTracker.ts';
import type { UnknownAgent } from '../env/UnknownAgent.ts';

/** Enemy team destroyed at least this much → the episode counts as a win. */
const WIN_THRESHOLD = 0.8;
/** Reward for a win (an average-contribution tank gets exactly this). */
const WIN_REWARD = 3;
/** Penalty at 0% enemy damage; fades linearly to 0 at WIN_THRESHOLD. */
const MAX_LOSS_PENALTY = 3;
/** Team-spirit τ: 0 = selfish, 1 = fully cooperative. Fixed for the MVP scenario. */
const TEAM_SPIRIT = 0.5;

/** Cumulative combat score for the tank's player (delta'd by the agent). */
export function calculateActionReward(eid: number, { world } = GameDI): number {
    const { PlayerRef } = getGameComponents(world);
    return scoreTracker.getScore(PlayerRef.id[eid]);
}

/**
 * Episode-end reward for `eid`. `getTeamDestroyedRatio` reports how much of a
 * team's initial health is gone, in [0, 1] (Scenario.getTeamDestroyedRatio).
 *
 * Win (enemy team ≥ WIN_THRESHOLD destroyed): WIN_REWARD weighted by the tank's
 * relative combat-score contribution, team-spirit blended — the tank that did the
 * killing earns more than one that just drove around, team average stays WIN_REWARD.
 * Otherwise: a penalty shared equally, MAX_LOSS_PENALTY at 0% enemy damage and
 * shrinking linearly to 0 as the team approaches the win threshold.
 */
export function calculateFinalReward(
    eid: number,
    getTeamDestroyedRatio: (teamId: number) => number,
    agents: UnknownAgent[],
    { world } = GameDI,
): number {
    const myTeam = getTankTeamId(eid);
    const enemyDestroyed = getTeamDestroyedRatio(1 - myTeam);
    const progress = clamp(enemyDestroyed / WIN_THRESHOLD, 0, 1);

    if (progress < 1) {
        return -MAX_LOSS_PENALTY * (1 - progress);
    }

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

    return WIN_REWARD * contribution;
}
