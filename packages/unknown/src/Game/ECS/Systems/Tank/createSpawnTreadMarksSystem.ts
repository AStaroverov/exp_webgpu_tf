import { query } from 'bitecs';
import { abs } from '../../../../../../../lib/math.ts';
import { random } from '../../../../../../../lib/random.ts';
import { spawnTreadMark, TreadMarkOptions } from '../../Entities/TreadMark.ts';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { Worlds } from '../../../DI/Worlds.ts';

const BASE_TREAD_MARK_CHANCE = 0.005;
const ANGVEL_MULTIPLIER = 0.025;
const MIN_SPEED = 0.5;
const MIN_ANGVEL = 0.1;

const treadMarkOptions: TreadMarkOptions = {
    x: 0,
    y: 0,
    width: 3,
    height: 5,
    rotation: 0,
};

export function createSpawnTreadMarksSystem({ physicsWorld } = Worlds) {
    const { VehiclePartCaterpillar, RigidBodyState } = getPhysicsWorldComponents(physicsWorld);

    return (_delta: number) => {
        const caterpillarEids = query(physicsWorld, [VehiclePartCaterpillar, RigidBodyState]);

        for (const eid of caterpillarEids) {
            const linvel = RigidBodyState.linvel.getBatch(eid);
            const angvel = RigidBodyState.angvel[eid];

            const speed = Math.sqrt(linvel[0] * linvel[0] + linvel[1] * linvel[1]);

            if (speed < MIN_SPEED && abs(angvel) < MIN_ANGVEL) {
                continue;
            }

            const turnBonus = abs(angvel) * ANGVEL_MULTIPLIER;
            const treadMarkChance = BASE_TREAD_MARK_CHANCE + turnBonus;

            if (random() > treadMarkChance) continue;

            // Caterpillar parts are slot-fill leaves (no brain node); the render mirrors
            // the physics body, so read the world transform from RigidBodyState directly.
            const pos = RigidBodyState.position.getBatch(eid);

            treadMarkOptions.x = pos[0];
            treadMarkOptions.y = pos[1];
            treadMarkOptions.rotation = RigidBodyState.rotation[eid];
            spawnTreadMark(treadMarkOptions);
        }
    };
}
