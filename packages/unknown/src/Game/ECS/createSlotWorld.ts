import { createWorld, World } from 'bitecs';
import { Opaque } from '../../../../renderer/src/type.ts';

import { createSlotComponent } from './Components/Slot.ts';
import { createColorComponent } from '../../../../renderer/src/ECS/Components/Common.ts';
import { createOccupantRefComponent } from '../DI/links.ts';

function createSlotOnlyComponents(world: World) {
    return {
        Slot: createSlotComponent(world),
        // Slot-owned (darkened) seed color captured at creation; applied to the part
        // that fills the slot. Own per-world Color instance (factory, not a singleton).
        Color: createColorComponent(world),
        OccupantRef: createOccupantRefComponent(world), // slot -> part atom (physics eid), 0 = empty
    };
}

export type SlotWorldComponents = ReturnType<typeof createSlotOnlyComponents>;

export type SlotWorld = Opaque<'SlotWorld', World<{
    components: SlotWorldComponents;
    time: {
        delta: number;
        elapsed: number;
        then: number;
    };
}>>;

export function createSlotWorld(): SlotWorld {
    const context = {
        components: null as unknown as SlotWorldComponents,
        time: {
            delta: 0,
            elapsed: 0,
            then: performance.now(),
        },
    };
    const world = createWorld(context) as unknown as SlotWorld;
    context.components = createSlotOnlyComponents(world);
    return world;
}

export function getSlotWorldComponents(world: SlotWorld): SlotWorldComponents {
    const components = world.components;
    if (!components) {
        throw new Error('Slot components are not available on this world');
    }
    return components;
}
