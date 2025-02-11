import { addComponent, defineComponent, Types } from 'bitecs';
import { World } from '../../../../../src/ECS/world.ts';

export const RigidBodyRef = defineComponent({
    id: Types.f64,
});

export function addRigidBodyRef(world: World, worldId: number, physicalId: number) {
    addComponent(world, RigidBodyRef, worldId);
    RigidBodyRef.id[worldId] = physicalId;
}
