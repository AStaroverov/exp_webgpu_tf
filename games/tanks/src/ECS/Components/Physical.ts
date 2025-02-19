import { addComponent } from 'bitecs';
import { DI } from '../../DI';
import { delegate } from '../../../../../src/delegate.ts';

const mapPhysicalIdToEntityId = new Map<number, number>();

export const RigidBodyRef = ({
    id: new Float64Array(delegate.defaultSize),
});

export function addRigidBodyRef(eid: number, pid: number, { world } = DI) {
    addComponent(world, eid, RigidBodyRef);
    RigidBodyRef.id[eid] = pid;
    mapPhysicalIdToEntityId.set(pid, eid);
}

export function resetRigidBodyRef(eid: number) {
    mapPhysicalIdToEntityId.delete(RigidBodyRef.id[eid]);
    RigidBodyRef.id[eid] = 0;
}

export function getEntityIdByPhysicalId(physicalId: number): number {
    if (!mapPhysicalIdToEntityId.has(physicalId)) throw new Error(`Entity with physicalId ${ physicalId } not found`);
    return mapPhysicalIdToEntityId.get(physicalId)!;
}