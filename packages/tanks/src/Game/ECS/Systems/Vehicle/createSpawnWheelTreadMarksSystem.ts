import { query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { Wheel } from '../../Components/Wheel.ts';
import { RigidBodyState } from '../../Components/Physical.ts';
import { Impulse } from '../../Components/Impulse.ts';
import { abs } from '../../../../../../../lib/math.ts';
import { random } from '../../../../../../../lib/random.ts';
import { spawnTreadMark } from '../../Entities/TreadMark.ts';
import {
    GlobalTransform,
    getMatrixTranslationX,
    getMatrixTranslationY,
    getMatrixRotationZ,
} from '../../../../../../renderer/src/ECS/Components/Transform.ts';

// Base chance
const BASE_CHANCE = 0.01;
// Chance when velocity doesn't align with impulse (skidding)
const SKID_CHANCE = 0.5;
// Chance multiplier for rotation
const ROTATION_MULTIPLIER = 0.3;

// Wheel tread mark dimensions
const TREAD_MARK_WIDTH = 2.5;
const TREAD_MARK_HEIGHT = 4;

export function createSpawnWheelTreadMarksSystem({ world } = GameDI) {
    return (_delta: number) => {
        const wheelEids = query(world, [Wheel, RigidBodyState, Impulse]);

        for (const eid of wheelEids) {
            const linvel = RigidBodyState.linvel.getBatch(eid);
            const rotation = RigidBodyState.rotation[eid];
            
            const impulseX = Impulse.x[eid];
            const impulseY = Impulse.y[eid];

            let chance = BASE_CHANCE;

            const dotProduct = linvel[0] * impulseX + linvel[1] * impulseY;
            // Negative or zero dotProduct = impulse against or perpendicular to velocity = skidding
            if (dotProduct <= 0) {
                chance = Math.max(chance, SKID_CHANCE);
            }

            // 2. Rotation increases chance
            if (rotation !== 0) {
                chance = Math.max(chance, abs(rotation) * ROTATION_MULTIPLIER);
            }

            // Random chance to spawn tread mark
            if (random() > chance) continue;

            const globalMatrix = GlobalTransform.matrix.getBatch(eid);
            const x = getMatrixTranslationX(globalMatrix);
            const y = getMatrixTranslationY(globalMatrix);
            const rot = getMatrixRotationZ(globalMatrix);

            spawnTreadMark({
                x,
                y,
                width: TREAD_MARK_WIDTH,
                height: TREAD_MARK_HEIGHT,
                rotation: rot,
            });
        }
    };
}
