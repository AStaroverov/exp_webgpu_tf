import { clamp } from 'lodash';
import { LEARNING_STEPS } from '../../../ml-common/consts.ts';
import { PlayerRef } from '../../../tanks/src/Game/ECS/Components/PlayerRef.ts';
import { Score } from '../../../tanks/src/Game/ECS/Components/Score.ts';

export const WEIGHTS = ({
    SCORE_POSITIVE_MUL: 1,
    SCORE_NEGATIVE_MUL: 2,
    // Death
    DEATH_MUL: 5,
    // Hit/Kill rewards
    KILL_REWARD: 5,
    HIT_REWARD: 0.5,
    // ML Score System rewards
    ADJACENT_ENEMY_REWARD: 2,
    EXPLORATION_WITH_ENEMY_REWARD: 0.25,
    EXPLORATION_WITHOUT_ENEMY_REWARD: 0.5,
    PROXIMITY_PENALTY: -0.05,
});

export const getFramePenalty = (frame: number) =>
    -clamp(Math.log10(1 + frame), 0, 3) / 100;

export const getDeathPenalty = (isDead: boolean) =>
    isDead ? -WEIGHTS.DEATH_MUL : 0;

export const getFinalReward = (successRatio: number, networkVersion: number) => 
    successRatio * (0.5 * (0.3 + 0.7 * clamp(networkVersion / LEARNING_STEPS, 0, 1)));

export function calculateActionReward(vehicleEid: number, weights: typeof WEIGHTS): number {
    const scoreReward = getTankScore(vehicleEid, weights);
    
    return (0
        + scoreReward
    );
}

function getTankScore(tankEid: number, weights: typeof WEIGHTS): number {
    const playerId = PlayerRef.id[tankEid];
    const score = (0
        + Score.positiveScore[playerId] * weights.SCORE_POSITIVE_MUL 
        + Score.negativeScore[playerId] * weights.SCORE_NEGATIVE_MUL
    );
    return score;
}