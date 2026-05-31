import { addEntity, EntityId } from 'bitecs';
import { getPhysicsWorldComponents, PhysicsWorld } from '../createPhysicsWorld.ts';

export function createPlayer(world: PhysicsWorld, teamId: number): EntityId {
    const { TeamRef } = getPhysicsWorldComponents(world);
    const playerId = addEntity(world);
    TeamRef.addComponent(world, playerId, teamId);
    return playerId;
}
