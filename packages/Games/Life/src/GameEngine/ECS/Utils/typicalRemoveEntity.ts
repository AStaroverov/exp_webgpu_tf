import { hasComponent, removeEntity } from 'bitecs';
import { Children } from '../Components/Children.js';
import { GameDI } from '../../DI/GameDI.js';
import { Destroy } from '../Components/Destroy.js';
import { Parent } from '../Components/Parent.js';

export function scheduleRemoveEntity(eid: number, recursive = true, { world } = GameDI) {
    Destroy.addComponent(world, eid, recursive);
}

export function typicalRemoveEntity(eid: number, disconnect = true, { world } = GameDI) {
    if (disconnect && hasComponent(world, eid, Parent) && hasComponent(world, Parent.id[eid], Children)) {
        Children.removeChild(Parent.id[eid], eid);
    }

    removeEntity(world, eid);
}

export function recursiveTypicalRemoveEntity(eid: number, isRoot = true, { world } = GameDI) {
    if (hasComponent(world, eid, Children)) {
        for (let i = 0; i < Children.entitiesCount[eid]; i++) {
            recursiveTypicalRemoveEntity(Children.entitiesIds.get(eid, i), false);
        }
        Children.entitiesCount[eid] = 0;
    }

    typicalRemoveEntity(eid, isRoot);
}
