import { addComponent, defineComponent, Types } from 'bitecs';
import { DI } from '../../DI';

export const Parent = defineComponent({
    id: Types.f64,
});

export function addParentComponent(entity: number, eid: number, { world } = DI) {
    addComponent(world, Parent, entity);
    Parent.id[entity] = eid;
}