import { GameDI } from '../../DI/GameDI.ts';
import { query } from 'bitecs';
import { DestroyByTimeout } from '../Components/Destroy.ts';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.ts';

export function createDestroyByTimeoutSystem({ world } = GameDI) {
    return (delta: number) => {
        const eids = query(world, [DestroyByTimeout]);

        for (let i = 0; i < eids.length; i++) {
            const eid = eids[i];

            DestroyByTimeout.updateTimeout(eid, delta);

            if (DestroyByTimeout.timeout[eid] <= 0) {
                scheduleRemoveEntity(eid);
            }
        }
    };
}
