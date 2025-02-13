import { addComponent, defineComponent, Types } from 'bitecs';
import { World } from '../../../../../src/ECS/world.ts';

const mapPhysicalIdToEntityId = new Map<number, number>();

export const RigidBodyRef = defineComponent({
    id: Types.f64,
});

export function addRigidBodyRef(world: World, worldId: number, physicalId: number) {
    addComponent(world, RigidBodyRef, worldId);
    RigidBodyRef.id[worldId] = physicalId;
    mapPhysicalIdToEntityId.set(physicalId, worldId);
}

export function getEntityIdByPhysicalId(physicalId: number): number {
    if (!mapPhysicalIdToEntityId.has(physicalId)) throw new Error(`Entity with physicalId ${ physicalId } not found`);
    return mapPhysicalIdToEntityId.get(physicalId)!;
}