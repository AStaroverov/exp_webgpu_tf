import { addComponent } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { delegate } from '../../../../../src/delegate.ts';

export const Parent = ({
    id: new Float64Array(delegate.defaultSize),
});

export function addParentComponent(entity: number, parentEid: number, { world } = GameDI) {
    addComponent(world, entity, Parent);
    Parent.id[entity] = parentEid;
}