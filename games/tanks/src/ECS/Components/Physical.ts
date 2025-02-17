import { addComponent, defineComponent, removeComponent, Types } from 'bitecs';
import { DI } from '../../DI';

const mapPhysicalIdToEntityId = new Map<number, number>();

export const RigidBodyRef = defineComponent({
    id: Types.f64, // pid
});

export function addRigidBodyRef(eid: number, pid: number, { world } = DI) {
    addComponent(world, RigidBodyRef, eid);
    RigidBodyRef.id[eid] = pid;
    mapPhysicalIdToEntityId.set(pid, eid);
}

export function removeRigidBodyRef(eid: number, { world } = DI) {
    removeComponent(world, RigidBodyRef, eid);
    mapPhysicalIdToEntityId.delete(RigidBodyRef.id[eid]);
    RigidBodyRef.id[eid] = 0;
}

export function getEntityIdByPhysicalId(physicalId: number): number {
    if (!mapPhysicalIdToEntityId.has(physicalId)) throw new Error(`Entity with physicalId ${ physicalId } not found`);
    return mapPhysicalIdToEntityId.get(physicalId)!;
}