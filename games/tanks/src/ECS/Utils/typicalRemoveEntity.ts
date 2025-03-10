import { hasComponent, removeEntity } from 'bitecs';
import { RigidBodyRef } from '../Components/Physical.ts';
import { removeRigidShape } from '../../Physical/createRigid.ts';
import { Children } from '../Components/Children.ts';
import { DI } from '../../DI';
import { Destroy } from '../Components/Destroy.ts';

export function scheduleRemoveEntity(eid: number, recursive = true, { world } = DI) {
    Destroy.addComponent(world, eid, recursive);
}

export function typicalRemoveEntity(eid: number, { world } = DI) {
    if (hasComponent(world, eid, RigidBodyRef)) {
        removeRigidShape(eid);
        RigidBodyRef.reset(eid);
    }

    removeEntity(world, eid);
}

export function recursiveTypicalRemoveEntity(eid: number, { world } = DI) {
    if (hasComponent(world, eid, Children)) {
        for (let i = 0; i < Children.entitiesCount[eid]; i++) {
            recursiveTypicalRemoveEntity(Children.entitiesIds.get(eid, i));
        }
        Children.entitiesCount[eid] = 0;
    }

    typicalRemoveEntity(eid);
}
