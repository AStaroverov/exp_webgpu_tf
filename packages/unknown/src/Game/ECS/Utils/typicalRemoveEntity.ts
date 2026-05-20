import { hasComponent, removeEntity } from 'bitecs';
import { removeRigidShape } from '../../Physical/createRigid.ts';
import { GameDI } from '../../DI/GameDI.ts';
import { getGameComponents } from '../createGameWorld.ts';

export function scheduleRemoveEntity(eid: number, recursive = true, { world } = GameDI) {
    const { Destroy } = getGameComponents(world);
    Destroy.addComponent(world, eid, recursive);
}

export function typicalRemoveEntity(eid: number, disconnect = true, { world } = GameDI) {
    const { RigidBodyRef, Parent, Children } = getGameComponents(world);
    if (hasComponent(world, eid, RigidBodyRef)) {
        removeRigidShape(eid);
        RigidBodyRef.clear(eid);
    }

    if (disconnect && hasComponent(world, eid, Parent) && hasComponent(world, Parent.id[eid], Children)) {
        Children.removeChild(Parent.id[eid], eid);
    }

    removeEntity(world, eid);
}

export function recursiveTypicalRemoveEntity(eid: number, isRoot = true, { world } = GameDI) {
    const { Children } = getGameComponents(world);
    if (hasComponent(world, eid, Children)) {
        for (let i = 0; i < Children.entitiesCount[eid]; i++) {
            recursiveTypicalRemoveEntity(Children.entitiesIds.get(eid, i), false);
        }
        Children.entitiesCount[eid] = 0;
    }

    typicalRemoveEntity(eid, isRoot);
}
