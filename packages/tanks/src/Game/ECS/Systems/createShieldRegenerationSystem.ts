import { query } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { Vehicle, VehicleType } from '../Components/Vehicle.ts';
import { Children } from '../Components/Children.ts';
import { VehicleTurret } from '../Components/VehicleTurret.ts';
import { Slot } from '../Components/Slot.ts';
import { SlotPartType } from '../Components/SlotConfig.ts';
import { RigidBodyState } from '../Components/Physical.ts';
import { PlayerRef } from '../Components/PlayerRef.ts';
import { TeamRef } from '../Components/TeamRef.ts';
import { fillSlot } from '../Entities/Vehicle/VehicleParts.ts';
import { mutatedVehicleOptions, resetOptions } from '../Entities/Vehicle/Options.ts';
import { isSlot, isSlotEmpty } from '../Utils/SlotUtils.ts';

// Shield regeneration interval in milliseconds
const SHIELD_REGEN_INTERVAL = 100; // 2 seconds per shield element

/**
 * System that regenerates shield parts for harvesters
 * Periodically fills empty shield slots on harvester turrets
 */
export function createShieldRegenerationSystem({ world } = GameDI) {
    const regenTimers = new Map<number, number>(); // turretEid -> lastRegenTime

    return (_delta: number) => {
        const currentTime = performance.now();

        // Query all turrets attached to vehicles
        const turretEids = query(world, [VehicleTurret]);

        for (let i = 0; i < turretEids.length; i++) {
            const turretEid = turretEids[i];
            const vehicleEid = VehicleTurret.vehicleEId[turretEid];

            // Only process harvesters
            if (Vehicle.type[vehicleEid] !== VehicleType.Harvester) continue;

            // Check regeneration timer for this turret
            const lastRegenTime = regenTimers.get(turretEid) ?? 0;
            if (currentTime - lastRegenTime < SHIELD_REGEN_INTERVAL) continue;

            // Find first empty shield slot on turret
            const emptyShieldSlotEid = findFirstEmptyShieldSlot(turretEid);
            if (emptyShieldSlotEid === null) continue;

            // Prepare options for filling the slot
            const options = resetOptions(mutatedVehicleOptions);
            options.playerId = PlayerRef.id[vehicleEid];
            options.teamId = TeamRef.id[vehicleEid];
            options.x = RigidBodyState.position.get(vehicleEid, 0);
            options.y = RigidBodyState.position.get(vehicleEid, 1);
            options.rotation = RigidBodyState.rotation[vehicleEid];

            // Fill the empty shield slot
            fillSlot(emptyShieldSlotEid, options);

            // Update regeneration timer
            regenTimers.set(turretEid, currentTime);
        }

        // Clean up timers for removed turrets
        for (const turretEid of regenTimers.keys()) {
            if (!turretEids.includes(turretEid)) {
                regenTimers.delete(turretEid);
            }
        }
    };
}

/**
 * Find first empty shield slot among children of parent (turret)
 */
function findFirstEmptyShieldSlot(parentEid: number): number | null {
    const childCount = Children.entitiesCount[parentEid];

    for (let i = 0; i < childCount; i++) {
        const childEid = Children.entitiesIds.get(parentEid, i);

        if (!isSlot(childEid)) continue;
        if (Slot.partType[childEid] !== SlotPartType.Shield) continue;
        if (!isSlotEmpty(childEid)) continue;

        return childEid;
    }

    return null;
}

