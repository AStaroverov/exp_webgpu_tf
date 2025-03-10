import { DI } from '../../DI';
import { query } from 'bitecs';
import { Destroy } from '../Components/Destroy.ts';
import { recursiveTypicalRemoveEntity, typicalRemoveEntity } from '../Utils/typicalRemoveEntity.ts';

export function createDestroySystem({ world } = DI) {
    return () => {
        const eids = query(world, [Destroy]);

        for (let i = 0; i < eids.length; i++) {
            const eid = eids[i];
            const recursive = Destroy.recursive[eid] === 1;
            recursive ? recursiveTypicalRemoveEntity(eid) : typicalRemoveEntity(eid);
        }
    };
}
