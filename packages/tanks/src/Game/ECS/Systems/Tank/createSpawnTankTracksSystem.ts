import { hasComponent, query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { Vehicle } from '../../Components/Vehicle.ts';
import { Children } from '../../Components/Children.ts';
import { VehiclePartCaterpillar } from '../../Components/VehiclePart.ts';
import { RigidBodyState } from '../../Components/Physical.ts';
import { abs, cos, PI, sin } from '../../../../../../../lib/math.ts';
import { random } from '../../../../../../../lib/random.ts';
import { spawnTankTrack } from '../../Entities/TankTrack.ts';
import {
    GlobalTransform,
    getMatrixTranslationX,
    getMatrixTranslationY,
    getMatrixRotationZ,
} from '../../../../../../renderer/src/ECS/Components/Transform.ts';
import { getSlotFillerEid, isSlot } from '../../Utils/SlotUtils.ts';

// Base chance for a caterpillar element to leave a track per frame
const BASE_TRACK_CHANCE = 0.01;
// How much angular velocity increases the chance
const ANGVEL_MULTIPLIER = 0.05;

export function createSpawnTankTracksSystem({ world } = GameDI) {
    // Track last spawn time per entity to avoid too many tracks
    let currentTime = 0;

    return (delta: number) => {
        currentTime += delta;

        const vehicleEids = query(world, [Vehicle]);

        for (const vehicleEid of vehicleEids) {
            if (random() > 0.5) continue;

            const childrenEids = Children.entitiesIds.getBatch(vehicleEid);
            const childrenCount = Children.entitiesCount[vehicleEid];
            const linvel = RigidBodyState.linvel.getBatch(vehicleEid);
            const angvel = RigidBodyState.angvel[vehicleEid];
            const vehicleRotation = RigidBodyState.rotation[vehicleEid];

            // Calculate speed
            const forwardX = cos(vehicleRotation - PI / 2);
            const forwardY = sin(vehicleRotation - PI / 2);
            const speed = abs(linvel[0] * forwardX + linvel[1] * forwardY);

            // Only spawn tracks when vehicle is moving
            if (speed < 0.5 && abs(angvel) < 0.1) {
                continue;
            }

            // Increase chance based on angular velocity (turning)
            const turnBonus = abs(angvel) * ANGVEL_MULTIPLIER;
            const trackChance = BASE_TRACK_CHANCE + turnBonus;

            for (let i = 0; i < childrenCount; i++) {
                const slotEid = childrenEids[i];
                if (!isSlot(slotEid)) continue;

                const childEid = getSlotFillerEid(slotEid);
                if (childEid === 0 || !hasComponent(world, childEid, VehiclePartCaterpillar)) continue;

                // Random chance to spawn track
                if (random() > trackChance) continue;

                // Get caterpillar element position from GlobalTransform
                const globalMatrix = GlobalTransform.matrix.getBatch(childEid);
                const x = getMatrixTranslationX(globalMatrix);
                const y = getMatrixTranslationY(globalMatrix);
                const rotation = getMatrixRotationZ(globalMatrix);

                // Get width/height from RigidBodyState or use defaults
                // Caterpillar tracks are typically small rectangles
                const trackWidth = 3;
                const trackHeight = 5;

                spawnTankTrack({
                    x,
                    y,
                    width: trackWidth,
                    height: trackHeight,
                    rotation,
                });
            }
        }
    };
}
