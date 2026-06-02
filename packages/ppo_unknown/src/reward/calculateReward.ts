/**
 * Reward shaping for ppo_unknown — ports the tanks `calculateReward` concept:
 * the step reward is the DELTA of a cumulative, event-based combat score
 * (`ScoreTracker`), not an hp count. Only the simplest combat events are scored
 * (hits + kills) — all physics/realtime shaping from tanks (aim alignment, turret
 * tracking, movement, approach, proximity) is intentionally removed.
 *
 *   calculateActionReward(eid) — cumulative score for the tank's player (tanks'
 *                                `Score.getTotalScore`); the agent subtracts the
 *                                previous value to get the per-macro-action reward.
 *   getFramePenalty(frame)     — tiny per-decision time cost.
 *   calculateFinalReward(...)  — episode outcome with OpenAI-Five team-spirit blend.
 */

import { clamp } from 'lodash';
import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';
import { getTankTeamId } from '../../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { scoreTracker } from './ScoreTracker.ts';
import type { UnknownAgent } from '../env/UnknownAgent.ts';

const FINAL_REWARD_SCALE = 1.0;
/** Team-spirit τ: 0 = selfish, 1 = fully cooperative. Fixed for the MVP scenario. */
const TEAM_SPIRIT = 0.5;

export function getFramePenalty(_frame: number): number {
    return -0.001;
}

/** Cumulative combat score for the tank's player (delta'd by the agent). */
export function calculateActionReward(eid: number, { world } = GameDI): number {
    const { PlayerRef } = getGameComponents(world);
    return scoreTracker.getScore(PlayerRef.id[eid]);
}

/**
 * Episode-end reward for `eid`. `successRatio` is team-0 perspective in [-1, 1].
 * Team-spirit blends the agent's own contribution share with an equal team split;
 * on a loss the blame is shared equally (don't punish the top performer most).
 */
export function calculateFinalReward(
    eid: number,
    successRatio: number,
    agents: UnknownAgent[],
    { world } = GameDI,
): number {
    const { PlayerRef } = getGameComponents(world);

    const myTeam = getTankTeamId(eid);
    const sign = myTeam === 0 ? 1 : -1;
    const outcome = sign * successRatio * FINAL_REWARD_SCALE;

    const teammates = agents.filter((a) => getTankTeamId(a.tankEid) === myTeam);
    const teamSize = Math.max(1, teammates.length);

    const myScore = scoreTracker.getScore(PlayerRef.id[eid]);
    const totalTeamScore = teammates.reduce(
        (sum, a) => sum + scoreTracker.getScore(PlayerRef.id[a.tankEid]),
        0,
    );
    const individualShare = totalTeamScore > 0
        ? clamp(myScore / totalTeamScore, 0, 1)
        : 1 / teamSize;
    const teamShare = 1 / teamSize;

    const blendedShare = outcome >= 0
        ? (1 - TEAM_SPIRIT) * individualShare + TEAM_SPIRIT * teamShare
        : teamShare;

    return outcome * blendedShare * teamSize;
}
