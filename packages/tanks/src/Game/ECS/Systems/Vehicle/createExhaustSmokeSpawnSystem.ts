import { hasComponent, query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { ExhaustPipe } from '../../Components/ExhaustPipe.ts';
import { Parent } from '../../Components/Parent.ts';
import { Vehicle } from '../../Components/Vehicle.ts';
import { RigidBodyState } from '../../Components/Physical.ts';
import { ExhaustSmokeOptions, spawnExhaustSmoke } from '../../Entities/ExhaustSmoke.ts';
import {
    GlobalTransform,
    getMatrixTranslationX,
    getMatrixTranslationY,
    getMatrixRotationZ,
} from '../../../../../../renderer/src/ECS/Components/Transform.ts';
import { random } from '../../../../../../../lib/random.ts';
import { hypot } from '../../../../../../../lib/math.ts';

// Increased emission when accelerating
const ACCELERATION_EMISSION_MULTI = 5;
// Smoke particle size range
const SMOKE_SIZE_MIN = 3;
const SMOKE_SIZE_MAX = 6;
// Smoke initial velocity
const SMOKE_VELOCITY_BASE = 15;
const SMOKE_VELOCITY_VARIANCE = 5;

const exhaustSmokeOptions: ExhaustSmokeOptions = {
    x: 0,
    y: 0,
    velocityX: 0,
    velocityY: 0,
    size: 0,
};

export function createExhaustSmokeSpawnSystem({ world } = GameDI) {
    return (delta: number) => {
        const pipeEids = query(world, [ExhaustPipe]);
        const deltaSeconds = delta / 1000;

        for (const pipeEid of pipeEids) {
            const vehicleEid = Parent.id[pipeEid];

            // Check if vehicle exists and has required components
            if (!hasComponent(world, vehicleEid, Vehicle)) continue;

            // Get vehicle state
            const vehicleMatrix = GlobalTransform.matrix.getBatch(vehicleEid);
            const vehicleX = getMatrixTranslationX(vehicleMatrix);
            const vehicleY = getMatrixTranslationY(vehicleMatrix);
            const vehicleRotation = getMatrixRotationZ(vehicleMatrix);

            // Get vehicle velocity for emission intensity
            let speed = 0;
            if (hasComponent(world, vehicleEid, RigidBodyState)) {
                const linvel = RigidBodyState.linvel.getBatch(vehicleEid);
                speed = hypot(linvel[0], linvel[1]);
            }

            // Adjust emission rate based on speed
            const speedFactor = Math.min(speed / 50, 1); // Normalize to 0-1
            const emissionRate = ExhaustPipe.emissionRate[pipeEid] * (1 + speedFactor * ACCELERATION_EMISSION_MULTI);

            // Accumulate time for emission
            ExhaustPipe.emissionAccumulator[pipeEid] += deltaSeconds * emissionRate;

            // Calculate world position of exhaust pipe
            const relX = ExhaustPipe.relativeX[pipeEid];
            const relY = ExhaustPipe.relativeY[pipeEid];
            const cos = Math.cos(vehicleRotation);
            const sin = Math.sin(vehicleRotation);

            const worldX = vehicleX + relX * cos - relY * sin;
            const worldY = vehicleY + relX * sin + relY * cos;

            // Calculate exhaust direction in world space
            const exhaustDir = vehicleRotation + ExhaustPipe.direction[pipeEid];
            
            // Spawn smoke particles based on accumulated time
            while (ExhaustPipe.emissionAccumulator[pipeEid] >= 1) {
                ExhaustPipe.emissionAccumulator[pipeEid] -= 1;

                const velocityMagnitude = SMOKE_VELOCITY_BASE + random() * SMOKE_VELOCITY_VARIANCE;
                const spread = (random() - 0.5) * 0.5; // ±0.25 radians spread
                const finalDir = exhaustDir + spread;

                const velocityX = Math.cos(finalDir) * velocityMagnitude;
                const velocityY = Math.sin(finalDir) * velocityMagnitude;

                // Random size variation
                const size = SMOKE_SIZE_MIN + random() * (SMOKE_SIZE_MAX - SMOKE_SIZE_MIN);

                exhaustSmokeOptions.x = worldX;
                exhaustSmokeOptions.y = worldY;
                exhaustSmokeOptions.size = size;
                exhaustSmokeOptions.velocityX = velocityX;
                exhaustSmokeOptions.velocityY = velocityY;
                spawnExhaustSmoke(exhaustSmokeOptions);
            }
        }
    };
}

