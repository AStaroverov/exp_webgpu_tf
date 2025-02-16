import { addRigidBodyRef } from './Physical.ts';
import { DI } from '../../DI';
import { addEntity } from 'bitecs';

export function createRigidGroup(
    pid: number,
    { world } = DI,
): [id: number, physicalId: number] {
    const eid = addEntity(world);
    addRigidBodyRef(world, eid, pid);
    return [eid, pid];
}
