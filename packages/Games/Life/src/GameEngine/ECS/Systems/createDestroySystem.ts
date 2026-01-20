import { GameDI } from '../../DI/GameDI.js';
import { query } from 'bitecs';
import { Destroy } from '../Components/Destroy.js';
import { recursiveTypicalRemoveEntity, typicalRemoveEntity } from '../Utils/typicalRemoveEntity.js';

export function createDestroySystem({ world } = GameDI) {
    return () => {
        const eids = query(world, [Destroy]);

        for (let i = 0; i < eids.length; i++) {
            const eid = eids[i];
            const recursive = Destroy.recursive[eid] === 1;
            recursive ? recursiveTypicalRemoveEntity(eid) : typicalRemoveEntity(eid);
        }
    };
}
