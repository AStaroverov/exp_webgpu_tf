import { query } from 'bitecs';
import { abs } from '../../../../../../../lib/math.ts';
import { random } from '../../../../../../../lib/random.ts';
import { spawnTreadMark } from '../../Entities/TreadMark.ts';
import {
    GlobalTransform,
    getMatrixTranslationX,
    getMatrixTranslationY,
    getMatrixRotationZ,
} from '../../../../../../renderer/src/ECS/Components/Transform.ts';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { getNodeByPhysics, getNodeRender } from '../../refs.ts';
import { Worlds } from '../../../DI/Worlds.ts';

const BASE_CHANCE = 0.01;
const SKID_CHANCE = 0.5;
const ROTATION_MULTIPLIER = 0.3;

const TREAD_MARK_WIDTH = 2.5;
const TREAD_MARK_HEIGHT = 4;

export function createSpawnWheelTreadMarksSystem({ physicsWorld } = Worlds) {
    const { Wheel, RigidBodyState, Impulse } = getPhysicsWorldComponents(physicsWorld);

    return (_delta: number) => {
        const wheelEids = query(physicsWorld, [Wheel, RigidBodyState, Impulse]);

        for (const eid of wheelEids) {
            const linvel = RigidBodyState.linvel.getBatch(eid);
            const rotation = RigidBodyState.rotation[eid];

            const impulseX = Impulse.x[eid];
            const impulseY = Impulse.y[eid];

            let chance = BASE_CHANCE;

            const dotProduct = linvel[0] * impulseX + linvel[1] * impulseY;
            if (dotProduct <= 0) {
                chance = Math.max(chance, SKID_CHANCE);
            }

            if (rotation !== 0) {
                chance = Math.max(chance, abs(rotation) * ROTATION_MULTIPLIER);
            }

            if (random() > chance) continue;

            // Wheel atoms are brain nodes (children of the hull node); resolve render via the node.
            const globalMatrix = GlobalTransform.matrix.getBatch(getNodeRender(getNodeByPhysics(eid)));
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
