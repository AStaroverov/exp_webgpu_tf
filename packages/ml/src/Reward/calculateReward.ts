import { clamp } from 'lodash';
import { PlayerRef } from '../../../tanks/src/Game/ECS/Components/PlayerRef.ts';
import { Score } from '../../../tanks/src/Game/ECS/Components/Score.ts';
import { TankAgent } from '../../../tanks/src/Plugins/Pilots/Agents/CurrentActorAgent.ts';
import { scenariosCount } from '../../../ml-common/Curriculum/createScenarioByCurriculumState.ts';

export const WEIGHTS = ({
    // Event-based step rewards (hit/kill tracked by other systems)
    KILL_REWARD: 1.2,
    HIT_REWARD: 0.05,
    // Episode outcome scale
    FINAL_REWARD_SCALE: 1.0,
});

export const getFramePenalty = (_frame: number) => -0.001;

/**
 * Step reward: score delta since last action (hit, kill, gotHit, friendlyFire).
 */
export function calculateActionReward(vehicleEid: number): number {
    const playerId = PlayerRef.id[vehicleEid];
    return Score.getTotalScore(playerId);
}

/**
 * Team Spirit (τ) — OpenAI Five inspired.
 * Blends individual and team-mean rewards:
 *   r_agent = (1 - τ) × r_individual + τ × r_team_mean
 *
 * τ grows with curriculum stage: early stages (1v1) → individual,
 * late stages (3v3, self-play) → cooperative.
 */
function getTeamSpirit(scenarioIndex: number): number {
    const maxTau = 0.8;
    return clamp(scenarioIndex / (scenariosCount - 1), 0, 1) * maxTau;
}

/**
 * Episode-end reward with team spirit blending.
 */
export function calculateFinalReward(
    vehicleEid: number,
    successRatio: number,
    pilots: TankAgent[],
    scenarioIndex: number,
): number {
    // successRatio is continuous [-1, 1]: positive = winning, negative = losing
    const outcomeReward = successRatio * WEIGHTS.FINAL_REWARD_SCALE;

    const teamSize = pilots.length;
    const tau = getTeamSpirit(scenarioIndex);

    // Individual contribution share
    const playerId = PlayerRef.id[vehicleEid];
    const myScore = Score.getPositiveScore(playerId) + Score.getNegativeScore(playerId) / 2;
    const totalTeamScore = pilots.reduce((sum, agent) => {
        const pid = PlayerRef.id[agent.tankEid];
        return sum + Score.getPositiveScore(pid) + Score.getNegativeScore(pid) / 2;
    }, 0);
    const individualShare = totalTeamScore > 0
        ? clamp(myScore / totalTeamScore, 0, 1)
        : 1 / teamSize;

    // Team-mean share (equal split)
    const teamShare = 1 / teamSize;

    // On loss: equal blame (teamShare only) — don't punish the best performer most
    // On win: blend individual contribution with team share
    const blendedShare = outcomeReward >= 0
        ? (1 - tau) * individualShare + tau * teamShare
        : teamShare;

    return outcomeReward * blendedShare * teamSize;
}
