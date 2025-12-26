import { clamp } from 'lodash';
import { abs, hypot, max, min, smoothstep } from '../../../../lib/math.ts';
import { HeuristicsData } from '../../../tanks/src/Game/ECS/Components/HeuristicsData.ts';
import { getTankHealth, getTankScore } from '../../../tanks/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { ENV_RAY_BUFFER, ENV_RAYS_TOTAL, RayHitType, TankInputTensor, TURRET_RAY_BUFFER } from '../../../tanks/src/Pilots/Components/TankState.ts';
import { LEARNING_STEPS } from '../../../ml-common/consts.ts';
import { TurretController } from '../../../tanks/src/Game/ECS/Components/TurretController.ts';
import { Tank } from '../../../tanks/src/Game/ECS/Components/Tank.ts';
import { VehicleController } from '../../../tanks/src/Game/ECS/Components/VehicleController.ts';

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
    MAP_BORDER_MULTIPLIER: 0.1,

    POSITIONING: {
        STACKING_PENALTY: -1,
        SEEKING_REWARD: 0.2,
    },
    POSITIONING_MULTIPLIER: 1,

    SHOOTING: {
        ENEMY_REWARD: 1,
        ALLY_PENALTY: -1,
        OBSTACLE_PENALTY: -0.3,
    },
    SHOOTING_MULTIPLIER: 0.3,
});

export function calculateActionReward(vehicleEid: number, width: number, height: number): number {
    const currentScore = getTankScore(vehicleEid);
    const currentHealth = getTankHealth(vehicleEid);
    // const [currentVehicleX, currentVehicleY] = RigidBodyState.position.getBatch(vehicleEid);

    const scoreReward = WEIGHTS.COMMON.SCORE * currentScore * 0.33;
    const healthReward = WEIGHTS.COMMON.HEALTH * currentHealth * 10;

    // const mapAwarenessReward = WEIGHTS.MAP_BORDER_MULTIPLIER * calculateVehicleMapAwarenessReward(
    //     width,
    //     height,
    //     currentVehicleX,
    //     currentVehicleY,
    // );

    return (0
        + scoreReward
        + healthReward
    );
}

export function calculateStateReward(vehicleEid: number, width: number, height: number): number {
    const envRaysBuffer = TankInputTensor.envRaysData.getBatch(vehicleEid);
    const turretRaysBuffer = TankInputTensor.turretRaysData.getBatch(vehicleEid);
    const speedBuffer = TankInputTensor.speed.getBatch(vehicleEid);
    const turretEid = Tank.turretEId[vehicleEid];
    const tankRotation = TankInputTensor.rotation[vehicleEid];
    const moveInput = VehicleController.move[vehicleEid];

    const envRayPositioningReward = WEIGHTS.POSITIONING_MULTIPLIER * calculateEnvRayPositioningReward(
        envRaysBuffer,
        HeuristicsData.approxColliderRadius[vehicleEid],
        tankRotation,
        moveInput
    );
    const seekingReward = WEIGHTS.POSITIONING_MULTIPLIER * calculateSeekingReward(
        envRaysBuffer,
        turretRaysBuffer,
        speedBuffer,
    );

    const shootingReward = WEIGHTS.SHOOTING_MULTIPLIER * calculateShootingReward(
        TurretController.shoot[turretEid],
        turretRaysBuffer,
    );

    
    return envRayPositioningReward + shootingReward + seekingReward;
}

function calculateShootingReward(isShooting: number, turretRaysBuffer: Float64Array): number {
    if (!isShooting) {
        return 0;
    }

    const hitType = turretRaysBuffer[TURRET_RAY_BUFFER * 0 + 2];

    switch (hitType) {
        case RayHitType.ENEMY_VEHICLE:
            return WEIGHTS.SHOOTING.ENEMY_REWARD;
        case RayHitType.ALLY_VEHICLE:
            return WEIGHTS.SHOOTING.ALLY_PENALTY;
        case RayHitType.OBSTACLE:
            return WEIGHTS.SHOOTING.OBSTACLE_PENALTY;
        case RayHitType.NONE:
        default:
            return 0;
    }
}

function calculateEnvRayPositioningReward(
    envRaysBuffer: Float64Array,
    colliderRadius: number,
    tankRotation: number,
    moveInput: number
): number {
    if (Math.abs(moveInput) < 0.1) {
        return 0;
    }
    
    let totalReward = 0;
    
    const forwardX = Math.cos(tankRotation);
    const forwardY = Math.sin(tankRotation);
    
    for (let i = 0; i < ENV_RAYS_TOTAL; i++) {
        const rayDirX = envRaysBuffer[i * ENV_RAY_BUFFER + 0];
        const rayDirY = envRaysBuffer[i * ENV_RAY_BUFFER + 1];
        const hitType = envRaysBuffer[i * ENV_RAY_BUFFER + 2];
        const distance = envRaysBuffer[i * ENV_RAY_BUFFER + 3];
        
        if (hitType === 0 || distance > colliderRadius * 1.5) continue;
        // Чем ближе к препятствию, тем сильнее штраф
        const proximityFactor = 1 - smoothstep(colliderRadius, colliderRadius * 1.5, distance);
        
        // Dot product между направлением танка и направлением к препятствию
        const forwardDot = forwardX * rayDirX + forwardY * rayDirY;
        
        // moveInput > 0: едем вперёд, проверяем переднее направление
        // moveInput < 0: едем назад, проверяем заднее направление (инвертируем dot)
        const effectiveDot = moveInput > 0 ? forwardDot : -forwardDot;
        
        // Штрафуем только если направление движения указывает на стенку (effectiveDot > 0)
        if (effectiveDot < 0) continue;

        const penalty = WEIGHTS.POSITIONING.STACKING_PENALTY 
            * abs(moveInput) 
            * effectiveDot 
            * proximityFactor;

        totalReward += penalty / ENV_RAYS_TOTAL;
    }

    return totalReward;
}


function calculateSeekingReward(
    envRaysBuffer: Float64Array,
    turretRaysBuffer: Float64Array,
    speedBuffer: Float64Array,
): number {
    // Проверяем есть ли противник в любом луче
    let hasEnemyInSight = false;

    // Проверяем env лучи
    for (let i = 0; i < ENV_RAYS_TOTAL; i++) {
        const hitType = envRaysBuffer[i * ENV_RAY_BUFFER + 2];
        if (hitType === RayHitType.ENEMY_VEHICLE) {
            hasEnemyInSight = true;
            break;
        }
    }

    // Проверяем turret луч
    if (!hasEnemyInSight) {
        const turretHitType = turretRaysBuffer[TURRET_RAY_BUFFER * 0 + 2];
        if (turretHitType === RayHitType.ENEMY_VEHICLE) {
            hasEnemyInSight = true;
        }
    }

    if (hasEnemyInSight) {
        return 0;
    }

    // Нет противников в поле зрения - поощряем движение для поиска
    const vx = speedBuffer[0];
    const vy = speedBuffer[1];
    const speed = hypot(vx, vy);

    if (speed < 10) {
        return 0;
    }

    // Нормализуем скорость (чем быстрее едем, тем больше награда, но с насыщением)
    const normalizedSpeed = smoothstep(0, 30, speed);

    return WEIGHTS.POSITIONING.SEEKING_REWARD * normalizedSpeed;
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
