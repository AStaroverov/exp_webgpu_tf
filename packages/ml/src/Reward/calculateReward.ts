import { clamp } from 'lodash';
import { hypot, max, min, smoothstep } from '../../../../lib/math.ts';
import { HeuristicsData } from '../../../tanks/src/Game/ECS/Components/HeuristicsData.ts';
import { RigidBodyState } from '../../../tanks/src/Game/ECS/Components/Physical.ts';
import { getTankHealth, getTankScore } from '../../../tanks/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { ENV_RAY_BUFFER, ENV_RAYS_TOTAL, TankInputTensor } from '../../../tanks/src/Pilots/Components/TankState.ts';
import { LEARNING_STEPS } from '../../../ml-common/consts.ts';

export const getFramePenalty = (frame: number) =>
    -clamp(Math.log10(1 + frame), 0, 3) / 100;

export const getFinalReward = (successRatio: number, networkVersion: number) => 
    successRatio * (0.5 * (0.3 + 0.7 * clamp(networkVersion / LEARNING_STEPS, 0, 1)));

const WEIGHTS = ({
    COMMON: {
        SCORE: 1,
        HEALTH: 1,
    },
    COMMON_MULTIPLIER: 1,

    MAP_BORDER: {
        PENALTY: -1,
    },
    MAP_BORDER_MULTIPLIER: 1,

    DISTANCE: {
        TOO_CLOSE_PENALTY: -1,
    },
    DISTANCE_MULTIPLIER: 0.5,
});

export function calculateReward(vehicleEid: number, width: number, height: number): number {
    const currentScore = getTankScore(vehicleEid);
    const currentHealth = getTankHealth(vehicleEid);
    const envRaysBuffer = TankInputTensor.envRaysData.getBatch(vehicleEid);
    const [currentVehicleX, currentVehicleY] = RigidBodyState.position.getBatch(vehicleEid);

    const scoreReward = WEIGHTS.COMMON.SCORE * currentScore * 0.33;
    const healthReward = WEIGHTS.COMMON.HEALTH * currentHealth * 10;

    const mapAwarenessReward = WEIGHTS.MAP_BORDER_MULTIPLIER * calculateVehicleMapAwarenessReward(
        width,
        height,
        currentVehicleX,
        currentVehicleY,
    );

    const envRayPositioningReward = WEIGHTS.DISTANCE_MULTIPLIER * calculateEnvRayPositioningReward(
        envRaysBuffer,
        HeuristicsData.approxColliderRadius[vehicleEid]
    );

    return (0
        + scoreReward
        + healthReward
        + mapAwarenessReward
        + envRayPositioningReward
    );
}

function calculateEnvRayPositioningReward(envRaysBuffer: Float64Array, colliderRadius: number): number {
    let maxPenalty = 0;
    
    for (let i = 0; i < ENV_RAYS_TOTAL; i++) {
        const hitType = envRaysBuffer[i * ENV_RAY_BUFFER + 2];
        const distance = envRaysBuffer[i * ENV_RAY_BUFFER + 6];
        
        if (hitType > 0 && distance < colliderRadius * 1.5) {
            const t = smoothstep(colliderRadius, colliderRadius * 1.5, distance);
            const penalty = WEIGHTS.DISTANCE.TOO_CLOSE_PENALTY * (1 - t);
            maxPenalty = min(maxPenalty, penalty);
        }
    }
    
    return maxPenalty;
}

function calculateVehicleMapAwarenessReward(
    width: number,
    height: number,
    x: number,
    y: number,
): number {
    const gapW = 0;
    const gapH = 0;
    const isInBounds = x >= gapW && x <= (width - gapW) && y >= gapH && y <= (height - gapH);

    if (isInBounds) {
        return 0;
    } else {
        const dist = distanceToMap(gapW, gapH, width - gapW, height - gapH, x, y);
        const score = WEIGHTS.MAP_BORDER.PENALTY * smoothstep(0, 100, dist);
        return score;
    }
}

function distanceToMap(
    minX: number, minY: number,
    maxX: number, maxY: number,
    x: number, y: number
): number {
    const closestX = max(minX, min(maxX, x));
    const closestY = max(minY, min(maxY, y));
    const dx = x - closestX;
    const dy = y - closestY;

    return hypot(dx, dy);
}
