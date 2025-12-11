import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { SlotPartType } from './SlotConfig.ts';
import { Children } from './Children.ts';

export const Slot = component({
    // Anchor position relative to parent (tank/turret)
    anchorX: TypedArray.f64(delegate.defaultSize),
    anchorY: TypedArray.f64(delegate.defaultSize),
    width: TypedArray.f64(delegate.defaultSize),
    height: TypedArray.f64(delegate.defaultSize),
    
    // Part type configuration for this slot (SlotPartType enum)
    partType: TypedArray.i8(delegate.defaultSize),

    addComponent(
        world: World,
        eid: EntityId,
        x: number,
        y: number,
        width: number,
        height: number,
        partType: SlotPartType,
    ): EntityId {
        addComponent(world, eid, Slot);

        Slot.anchorX[eid] = x;
        Slot.anchorY[eid] = y;
        Slot.width[eid] = width;
        Slot.height[eid] = height;
        Slot.partType[eid] = partType;
        
        return eid;
    },

    /**
     * Check if slot has a filler (TankPart child)
     */
    isFilled(eid: EntityId): boolean {
        return Children.entitiesCount[eid] > 0;
    },

    /**
     * Check if slot is empty (no TankPart child)
     */
    isEmpty(eid: EntityId): boolean {
        return Children.entitiesCount[eid] === 0;
    },

    /**
     * Get filler entity ID (first child) or 0 if empty
     */
    getFillerEid(eid: EntityId): EntityId {
        if (Children.entitiesCount[eid] === 0) return 0;
        return Children.entitiesIds.get(eid, 0);
    },
});
