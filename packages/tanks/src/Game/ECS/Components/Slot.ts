import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { SlotPartType } from './SlotConfig.ts';

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

        Slot.partType[eid] = partType;
        Slot.anchorX[eid] = x;
        Slot.anchorY[eid] = y;
        Slot.width[eid] = width;
        Slot.height[eid] = height;
        
        return eid;
    },
});
