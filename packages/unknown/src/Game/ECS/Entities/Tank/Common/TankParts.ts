// Re-export from Vehicle for backward compatibility
export {
    createRectangleSet,
    createSlotEntities,
    fillAllSlots,
    fillSlot,
    getEmptySlotsCount,
    findFirstEmptySlot,
    getSlotCount,
    getFilledSlotCount,
    getVehicleTotalSlotCount as getTankTotalSlotCount,
    getVehicleFilledSlotCount as getTankFilledSlotCount,
    type PartsData,
} from '../../Vehicle/VehicleParts.ts';
