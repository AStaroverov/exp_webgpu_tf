import { EntityId, hasComponent, query } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { Spice } from '../Components/Spice.ts';
import { Debris } from '../Components/Debris.ts';
import { SpiceCollector } from '../Components/SpiceCollector.ts';
import { getEntityIdByPhysicalId } from '../Components/Physical.ts';
import { TeamRef } from '../Components/TeamRef.ts';
import { Score } from '../Components/Score.ts';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.ts';
import { Shape } from 'renderer/src/ECS/Components/Shape.ts';

/**
 * Find player entity by team ID
 */
function findPlayerByTeam(teamId: number, world = GameDI.world): EntityId | null {
    const players = query(world, [Score, TeamRef]);
    for (const playerEid of players) {
        if (TeamRef.id[playerEid] === teamId) {
            return playerEid;
        }
    }
    return null;
}

/**
 * Handler for collector collision events (spice and debris).
 * Called from createGame when collision events are processed.
 */
function handleCollectorCollision(
    eid1: EntityId,
    eid2: EntityId,
    started: boolean,
    world = GameDI.world
) {
    if (!started) return; // Only handle collision start

    // Determine which is the collectible and which is the collector
    let collectibleEid: EntityId | null = null;
    let collectorEid: EntityId | null = null;
    let isSpice = false;
    let isDebris = false;

    // Check for Spice
    if (hasComponent(world, eid1, Spice) && hasComponent(world, eid2, SpiceCollector)) {
        collectibleEid = eid1;
        collectorEid = eid2;
        isSpice = true;
    } else if (hasComponent(world, eid2, Spice) && hasComponent(world, eid1, SpiceCollector)) {
        collectibleEid = eid2;
        collectorEid = eid1;
        isSpice = true;
    }
    // Check for Debris
    else if (hasComponent(world, eid1, Debris) && hasComponent(world, eid2, SpiceCollector)) {
        collectibleEid = eid1;
        collectorEid = eid2;
        isDebris = true;
    } else if (hasComponent(world, eid2, Debris) && hasComponent(world, eid1, SpiceCollector)) {
        collectibleEid = eid2;
        collectorEid = eid1;
        isDebris = true;
    }

    if (collectibleEid === null || collectorEid === null) return;

    // Get team from collector and find player
    const teamId = TeamRef.id[collectorEid];
    const playerEid = findPlayerByTeam(teamId, world);
    if (playerEid === null) return;

    if (isSpice) {
        const spiceValue = Shape.values.get(collectibleEid, 0);
        Score.addSpice(playerEid, spiceValue);
        scheduleRemoveEntity(collectibleEid);
    } else if (isDebris) {
        Score.addDebris(playerEid, 1);
        scheduleRemoveEntity(collectibleEid);
    }
}

/**
 * Creates a collision event handler that should be called from eventQueue.drainCollisionEvents
 */
export function createCollectorCollisionHandler({ world, physicalWorld } = GameDI) {
    return (handle1: number, handle2: number, started: boolean) => {
        const collider1 = physicalWorld.getCollider(handle1);
        const collider2 = physicalWorld.getCollider(handle2);
        
        if (!collider1 || !collider2) return;
        
        const rb1 = collider1.parent();
        const rb2 = collider2.parent();
        
        if (!rb1 || !rb2) return;
        
        const eid1 = getEntityIdByPhysicalId(rb1.handle);
        const eid2 = getEntityIdByPhysicalId(rb2.handle);
        
        if (eid1 === 0 || eid2 === 0) return;
        
        handleCollectorCollision(eid1, eid2, started, world);
    };
}
