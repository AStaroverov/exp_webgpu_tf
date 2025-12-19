import { query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { VehiclePartCaterpillar } from '../../Components/VehiclePart.ts';
import { RigidBodyState } from '../../Components/Physical.ts';
import { abs } from '../../../../../../../lib/math.ts';
import { random } from '../../../../../../../lib/random.ts';
import { spawnTreadMark, TreadMarkOptions } from '../../Entities/TreadMark.ts';
import {
    GlobalTransform,
    getMatrixTranslationX,
    getMatrixTranslationY,
    getMatrixRotationZ,
} from '../../../../../../renderer/src/ECS/Components/Transform.ts';

// Base chance for a caterpillar element to leave a tread mark per frame
const BASE_TREAD_MARK_CHANCE = 0.005;
// How much angular velocity increases the chance
const ANGVEL_MULTIPLIER = 0.025;
// Minimum speed to spawn tread marks
const MIN_SPEED = 0.5;
const MIN_ANGVEL = 0.1;

const treadMarkOptions: TreadMarkOptions = {
    x: 0,
    y: 0,
    width: 3,
    height: 5,
    rotation: 0,
};

export function createSpawnTreadMarksSystem({ world } = GameDI) {
    return (_delta: number) => {
        const caterpillarEids = query(world, [VehiclePartCaterpillar, RigidBodyState]);

        for (const eid of caterpillarEids) {
            const linvel = RigidBodyState.linvel.getBatch(eid);
            const angvel = RigidBodyState.angvel[eid];

            // Calculate speed from linear velocity
            const speed = Math.sqrt(linvel[0] * linvel[0] + linvel[1] * linvel[1]);

            // Only spawn tread marks when caterpillar is moving
            if (speed < MIN_SPEED && abs(angvel) < MIN_ANGVEL) {
                continue;
            }

            // Increase chance based on angular velocity (turning)
            const turnBonus = abs(angvel) * ANGVEL_MULTIPLIER;
            const treadMarkChance = BASE_TREAD_MARK_CHANCE + turnBonus;

            // Random chance to spawn tread mark
            if (random() > treadMarkChance) continue;

            // Get caterpillar element position from GlobalTransform
            const globalMatrix = GlobalTransform.matrix.getBatch(eid);
            const x = getMatrixTranslationX(globalMatrix);
            const y = getMatrixTranslationY(globalMatrix);
            const rotation = getMatrixRotationZ(globalMatrix);

            // Caterpillar tread marks are typically small rectangles
            
            treadMarkOptions.x = x;
            treadMarkOptions.y = y;
            treadMarkOptions.rotation = rotation;
            spawnTreadMark(treadMarkOptions);
        }
    };
}

