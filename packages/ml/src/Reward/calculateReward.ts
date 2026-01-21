import { clamp } from 'lodash';
import { PlayerRef } from '../../../tanks/src/Game/ECS/Components/PlayerRef.ts';
import { Score } from '../../../tanks/src/Game/ECS/Components/Score.ts';
import { TankAgent } from '../../../tanks/src/Plugins/Pilots/Agents/CurrentActorAgent.ts';

export const WEIGHTS = ({
    FINAL_REWARD_POOL: 1,
    // Hit/Kill rewards
    KILL_REWARD: 1.2,
    HIT_REWARD: 0.05,
    // ML Score System rewards
    ADJACENT_ENEMY_REWARD: 0.3,
    EXPLORATION_WITH_ENEMY_REWARD: 0.06,
    EXPLORATION_WITHOUT_ENEMY_REWARD: 0.02,
    PROXIMITY_PENALTY: 0.005,
});

export const getFramePenalty = (frame: number) => 0;
    // -clamp(Math.log10(1 + frame), 0, 3) / 100;

export function calculateActionReward(vehicleEid: number): number {
    const scoreReward = getTankScore(vehicleEid);
    
    return (0
        + scoreReward
    );
}

export function calculateFinalReward(vehicleEid: number, successRatio: number, pilots: TankAgent[]): number {
    if (successRatio <= 0.1) return 0;

    const totalTeamScore = pilots.reduce((sum, agent) => {
        const playerId = PlayerRef.id[agent.tankEid];
        return sum + Score.getPositiveScore(playerId) + Score.getNegativeScore(playerId) / 2;
    }, 0);

    if (totalTeamScore <= 0) return 0;

    const playerId = PlayerRef.id[vehicleEid];
    const myScore = Score.getPositiveScore(playerId) + Score.getNegativeScore(playerId) / 2;
    const proportion = clamp(myScore / totalTeamScore, 0, 1);
    
    return successRatio * proportion * (WEIGHTS.FINAL_REWARD_POOL * pilots.length);
}

function getTankScore(tankEid: number): number {
    const playerId = PlayerRef.id[tankEid];
    return Score.getTotalScore(playerId);
}