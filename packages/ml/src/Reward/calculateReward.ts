import { clamp } from 'lodash';
import { PlayerRef } from '../../../tanks/src/Game/ECS/Components/PlayerRef.ts';
import { Score } from '../../../tanks/src/Game/ECS/Components/Score.ts';
import { TankAgent } from '../../../tanks/src/Plugins/Pilots/Agents/CurrentActorAgent.ts';

export const WEIGHTS = ({
    WIN_REWARD: 3,
    FINAL_REWARD_POOL: 5,
    // Hit/Kill rewards
    KILL_REWARD: 5,
    HIT_REWARD: 0.2,
    // ML Score System rewards
    ADJACENT_ENEMY_REWARD: 4,
    EXPLORATION_WITH_ENEMY_REWARD: 0.1,
    EXPLORATION_WITHOUT_ENEMY_REWARD: 0.2,
    PROXIMITY_PENALTY: 0.05,
});

export const getFramePenalty = (frame: number) =>
    -clamp(Math.log10(1 + frame), 0, 3) / 100;

export function calculateActionReward(vehicleEid: number, weights: typeof WEIGHTS): number {
    const scoreReward = getTankScore(vehicleEid, weights);
    
    return (0
        + scoreReward
    );
}

export function calculateFinalReward(vehicleEid: number, successRatio: number, pilots: TankAgent[]): number {
    return 0;
    // if (successRatio <= 0.1) return 0;

    // const totalTeamScore = pilots.reduce((sum, agent) => {
    //     const playerId = PlayerRef.id[agent.tankEid];
    //     return sum + Score.positiveScore[playerId] + Score.negativeScore[playerId];
    // }, 0);

    // if (totalTeamScore <= 0) return 0;

    // const playerId = PlayerRef.id[vehicleEid];
    // const myScore = Score.positiveScore[playerId] + Score.negativeScore[playerId];
    // const proportion = clamp(myScore / totalTeamScore, 0, 1);
    
    // return WEIGHTS.WIN_REWARD + proportion * (WEIGHTS.FINAL_REWARD_POOL * pilots.length);
}

function getTankScore(tankEid: number, weights: typeof WEIGHTS): number {
    const playerId = PlayerRef.id[tankEid];
    const score = (0
        + Score.positiveScore[playerId]
        + Score.negativeScore[playerId]
    );
    return score;
}