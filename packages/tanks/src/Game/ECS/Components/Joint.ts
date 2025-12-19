import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { addComponent, EntityId, removeComponent, World } from 'bitecs';

/**
 * Component for entities connected to a parent via physical joint.
 * Stores the joint handle (pid) from the physics engine (Rapier).
 */
export const Joint = component({
    // Physical joint handle connecting this entity to its parent
    pid: TypedArray.f64(delegate.defaultSize),

    addComponent(world: World, eid: EntityId, pid: number): void {
        addComponent(world, eid, Joint);
        Joint.pid[eid] = pid;
    },

    removeComponent(world: World, eid: EntityId): void {
        removeComponent(world, eid, Joint);
        Joint.pid[eid] = 0;
    },

    setPid(eid: EntityId, pid: number): void {
        Joint.pid[eid] = pid;
    },

    resetComponent(eid: EntityId): void {
        Joint.pid[eid] = 0;
    },
});

