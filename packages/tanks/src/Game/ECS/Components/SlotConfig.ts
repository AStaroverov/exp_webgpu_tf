/**
 * Slot Configuration Component
 * 
 * Re-exports from centralized Config for backward compatibility.
 */

import {
    SlotPartType,
    getSlotPartConfig,
    SlotPartPhysics,
} from '../../Config/index.ts';

export { SlotPartType, getSlotPartConfig };

/**
 * Configuration for a slot part - immutable preset data
 */
export type SlotPartConfig = SlotPartPhysics & { density: number };
