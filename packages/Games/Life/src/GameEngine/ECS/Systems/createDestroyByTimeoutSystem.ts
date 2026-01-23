import { GameDI } from '../../DI/GameDI.js';
import { query } from 'bitecs';
import { DestroyByTimeout } from '../Components/Destroy.js';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.js';

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
