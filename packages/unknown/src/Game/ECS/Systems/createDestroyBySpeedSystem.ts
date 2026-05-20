import { GameDI } from '../../DI/GameDI.ts';
import { query } from 'bitecs';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.ts';
import { hypot } from '../../../../../../lib/math.ts';
import { getGameComponents } from '../createGameWorld.ts';

export function createDestroyBySpeedSystem({ world } = GameDI) {
    const { DestroyBySpeed, RigidBodyState } = getGameComponents(world);

    return () => {
        const eids = query(world, [DestroyBySpeed, RigidBodyState]);

        for (let i = 0; i < eids.length; i++) {
            const eid = eids[i];
            const linvel = RigidBodyState.linvel.getBatch(eid);
            const speed = hypot(linvel[0], linvel[1]);
            const minSpeed = DestroyBySpeed.minSpeed[eid];

            if (speed < minSpeed) {
                scheduleRemoveEntity(eid);
            }
        }
    };
}
