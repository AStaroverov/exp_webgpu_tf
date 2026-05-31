import { query } from 'bitecs';
import { recursiveTypicalRemoveEntity, typicalRemoveEntity } from '../Utils/typicalRemoveEntity.ts';
import { getPhysicsWorldComponents } from '../createPhysicsWorld.ts';
import { Worlds } from '../../DI/Worlds.ts';

export function createDestroySystem({ physicsWorld } = Worlds) {
    const { Destroy } = getPhysicsWorldComponents(physicsWorld);

    return () => {
        const eids = query(physicsWorld, [Destroy]);

        for (let i = 0; i < eids.length; i++) {
            const eid = eids[i];
            const recursive = Destroy.recursive[eid] === 1;
            recursive ? recursiveTypicalRemoveEntity(eid) : typicalRemoveEntity(eid);
        }
    };
}
