import { GameDI } from '../../DI/GameDI.ts';
import { query } from 'bitecs';
import { DestroyBySpeed } from '../Components/Destroy.ts';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.ts';
import { RigidBodyState } from '../Components/Physical.ts';
import { hypot } from '../../../../../../lib/math.ts';

export function createDestroyBySpeedSystem({ world } = GameDI) {
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
