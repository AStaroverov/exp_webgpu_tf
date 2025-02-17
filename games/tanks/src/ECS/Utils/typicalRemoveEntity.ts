import { hasComponent, removeEntity } from 'bitecs';
import { removeRigidBodyRef, RigidBodyRef } from '../Components/Physical.ts';
import { removeRigidShape } from '../../Physical/createRigid.ts';
import { Children } from '../Components/Children.ts';
import { DI } from '../../DI';

export function typicalRemoveEntity(eid: number, { world } = DI) {
    if (hasComponent(world, RigidBodyRef, eid)) {
        removeRigidShape(eid);
        removeRigidBodyRef(eid);
    }

    removeEntity(world, eid);
}

export function recursiveTypicalRemoveEntity(eid: number, { world } = DI) {
    if (hasComponent(world, Children, eid)) {
        const children = Children.entitiesIds[eid];
        for (let i = 0; i < Children.entitiesCount[eid]; i++) {
            typicalRemoveEntity(children[i]);
        }
        Children.entitiesCount[eid] = 0;
    }

    typicalRemoveEntity(eid);
}