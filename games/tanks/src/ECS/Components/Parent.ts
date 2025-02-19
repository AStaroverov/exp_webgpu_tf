import { addComponent } from 'bitecs';
import { DI } from '../../DI';
import { delegate } from '../../../../../src/delegate.ts';

export const Parent = ({
    id: new Float64Array(delegate.defaultSize),
});

export function addParentComponent(entity: number, eid: number, { world } = DI) {
    addComponent(world, entity, Parent);
    Parent.id[entity] = eid;
}