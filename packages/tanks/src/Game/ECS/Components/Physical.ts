import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { NestedArray, TypedArray } from '../../../../../renderer/src/utils.ts';

const mapPhysicalIdToEntityId = new Map<number, number>();

export const RigidBodyRef = component({
    id: new Float64Array(delegate.defaultSize),

    addComponent: (world: World, eid: number, pid: number) => {
        addComponent(world, eid, RigidBodyRef);
        RigidBodyRef.id[eid] = pid;
        mapPhysicalIdToEntityId.set(pid, eid);
    },

    clear: (eid: number) => {
        const pid = RigidBodyRef.id[eid];
        if (pid !== 0) {
            RigidBodyRef.id[eid] = 0;
            mapPhysicalIdToEntityId.delete(pid);
        }
    },

    dispose: () => {
        mapPhysicalIdToEntityId.clear();
    },
});

export function getEntityIdByPhysicalId(physicalId: number): number {
    if (!mapPhysicalIdToEntityId.has(physicalId)) throw new Error(`Entity with physicalId ${ physicalId } not found`);
    return mapPhysicalIdToEntityId.get(physicalId)!;
}

export const RigidBodyState = component({
    position: NestedArray.f64(2, delegate.defaultSize),
    rotation: TypedArray.f64(delegate.defaultSize),
    linvel: NestedArray.f64(2, delegate.defaultSize),
    angvel: TypedArray.f64(delegate.defaultSize),
    // mass: new Float64Array(delegate.defaultSize),
    // force: new Float64Array(delegate.defaultSize),
    // torque: new Float64Array(delegate.defaultSize),
    // invMass: new Float64Array(delegate.defaultSize),
    // inertia: new Float64Array(delegate.defaultSize),
    // invInertia: new Float64Array(delegate.defaultSize),
    // damping: new Float64Array(delegate.defaultSize),
    // restitution: new Float64Array(delegate.defaultSize),
    // friction: new Float64Array(delegate.defaultSize),
    // collisionGroups: new Uint32Array(delegate.defaultSize),
    // collisionGroupsMask: new Uint32Array(delegate.defaultSize),

    addComponent: (world: World, eid: EntityId) => {
        addComponent(world, eid, RigidBodyState);
    },

    update: (
        eid: number,
        translation: { x: number, y: number },
        rotation: number,
        linvel: { x: number, y: number },
        angvel: number,
    ) => {
        RigidBodyState.position.set(eid, 0, translation.x);
        RigidBodyState.position.set(eid, 1, translation.y);
        RigidBodyState.rotation[eid] = rotation;
        RigidBodyState.linvel.set(eid, 0, linvel.x);
        RigidBodyState.linvel.set(eid, 1, linvel.y);
        RigidBodyState.angvel[eid] = angvel;
    },
});