import { clamp } from 'lodash';
import { getTankHealth } from '../../../tanks/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { LEARNING_STEPS } from '../../../ml-common/consts.ts';
import { PlayerRef } from '../../../tanks/src/Game/ECS/Components/PlayerRef.ts';
import { Score } from '../../../tanks/src/Game/ECS/Components/Score.ts';

export const getFramePenalty = (frame: number) =>
    -clamp(Math.log10(1 + frame), 0, 3) / 100;

export const getFinalReward = (successRatio: number, networkVersion: number) => 
    successRatio * (0.5 * (0.3 + 0.7 * clamp(networkVersion / LEARNING_STEPS, 0, 1)));

// add randomised weights
export const WEIGHTS = ({
    HEALTH_MUL: 20,
    SCORE_POSITIVE_MUL: 0.33,
    SCORE_NEGATIVE_MUL: 0.66,
    DETECT_ENEMY_MUL: 3,
    BYPASS_MUL: 3,
});
// export const getRandomWeights = (): typeof WEIGHTS => {
//     const health = 0.5 + randomRangeFloat(0, 0.5);
//     const scorePositive = randomRangeFloat(0.2, 0.8);
//     const scoreNegative = 1 - scorePositive;
//     const stacking = 0.5 + randomRangeFloat(0, 0.5);
//     const detectEnemy = 0.7 + randomRangeFloat(0, 0.3);

//     return {
//         HEALTH_MUL: WEIGHTS.HEALTH_MUL * health,
//         SCORE_POSITIVE_MUL: WEIGHTS.SCORE_POSITIVE_MUL * scorePositive,
//         SCORE_NEGATIVE_MUL: WEIGHTS.SCORE_NEGATIVE_MUL * scoreNegative,
//         DETECT_ENEMY_MUL: WEIGHTS.DETECT_ENEMY_MUL * detectEnemy,
//     }
// }

export function calculateActionReward(vehicleEid: number, weights: typeof WEIGHTS): number {
    const scoreReward = getTankScore(vehicleEid, weights);
    const healthReward = weights.HEALTH_MUL * getTankHealth(vehicleEid);
    
    return (0
        + scoreReward
        + healthReward
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


// function calculateRayDetectEnemyReward(raysBuffer: Float64Array,): number {
//     let hitsCount = 0;
//     for (let i = 0; i < RAYS_COUNT; i++) {
//         const hitType = raysBuffer[i * RAY_BUFFER + 0];
//         if (hitType === RayHitType.ENEMY_VEHICLE) hitsCount++;
//         if (hitsCount >= 4) break;
//     }

//     return hitsCount > 1 ? hitsCount / 4 : 0;
// }