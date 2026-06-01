import { addEntity, EntityId } from 'bitecs';
import { getBrainWorldComponents, BrainWorld } from '../createBrainWorld.ts';

// Dead-but-must-compile: a standalone player entity carrying the canonical team ref,
// re-homed to the brain world in Step 3 (no live caller).
export function createPlayer(world: BrainWorld, teamId: number): EntityId {
    const { TeamRef } = getBrainWorldComponents(world);
    const playerId = addEntity(world);
    TeamRef.addComponent(world, playerId, teamId);
    return playerId;
}
