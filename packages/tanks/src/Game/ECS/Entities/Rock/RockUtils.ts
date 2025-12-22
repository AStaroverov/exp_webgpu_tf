import { EntityId, hasComponent } from 'bitecs';
import { randomRangeFloat } from '../../../../../../../lib/random.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { Children } from '../../Components/Children.ts';
import { DestroyByTimeout } from '../../Components/Destroy.ts';
import { Parent } from '../../Components/Parent.ts';
import { RockPart } from '../../Components/Rock.ts';
import { scheduleRemoveEntity } from '../../Utils/typicalRemoveEntity.ts';

/**
 * Mark a rock part for destruction.
 * Since rock parts are Fixed bodies, they just get marked as debris and scheduled for removal.
 */
export function tearOffRockPart(rockPartEid: EntityId, shouldBreakConnection: boolean = true, { world } = GameDI) {
    const parentEid = Parent.id[rockPartEid];

    if (shouldBreakConnection && parentEid > 0) {
        Children.removeChild(parentEid, rockPartEid);
    }

    if (hasComponent(world, rockPartEid, RockPart)) {
        RockPart.removeComponent(world, rockPartEid);
    }

    if (!hasComponent(world, rockPartEid, DestroyByTimeout)) {
        DestroyByTimeout.addComponent(world, rockPartEid, 3_000 + randomRangeFloat(0, 3_000));
    }
}

/**
 * Destroy a rock completely - remove all remaining parts
 */
export function destroyRock(rockEid: EntityId, { world } = GameDI) {
    // Collect all parts before tearing them off
    const partsToRemove: EntityId[] = [];
    const childCount = Children.entitiesCount[rockEid];

    for (let i = 0; i < childCount; i++) {
        const partEid = Children.entitiesIds.get(rockEid, i);
        if (hasComponent(world, partEid, RockPart)) {
            partsToRemove.push(partEid);
        }
    }

    // Tear off all parts
    for (const partEid of partsToRemove) {
        tearOffRockPart(partEid);
    }

    // Schedule removal of the rock entity
    scheduleRemoveEntity(rockEid, false);
}

