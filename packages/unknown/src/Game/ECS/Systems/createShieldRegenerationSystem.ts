import { query } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { VehicleType } from '../Components/Vehicle.ts';
import { SlotPartType } from '../Components/SlotConfig.ts';
import { fillSlot } from '../Entities/Vehicle/VehicleParts.ts';
import { mutatedVehicleOptions, resetOptions } from '../Entities/Vehicle/Options.ts';
import { isSlot, isSlotEmpty } from '../Utils/SlotUtils.ts';
import { ShieldConfig } from '../../Config/index.ts';
import { getGameComponents } from '../createGameWorld.ts';

const SHIELD_REGEN_INTERVAL = ShieldConfig.regenInterval;

export function createShieldRegenerationSystem({ world } = GameDI) {
    const { Vehicle, VehicleTurret, Parent, Slot, RigidBodyState, PlayerRef, TeamRef, Children } = getGameComponents(world);
    const regenTimers = new Map<number, number>();

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

    return (_delta: number) => {
        const currentTime = performance.now();

        const turretEids = query(world, [VehicleTurret]);

        for (let i = 0; i < turretEids.length; i++) {
            const turretEid = turretEids[i];
            const vehicleEid = Parent.id[turretEid];

            if (Vehicle.type[vehicleEid] !== VehicleType.Harvester) continue;

            const lastRegenTime = regenTimers.get(turretEid) ?? 0;
            if (currentTime - lastRegenTime < SHIELD_REGEN_INTERVAL) continue;

            const emptyShieldSlotEid = findFirstEmptyShieldSlot(turretEid);
            if (emptyShieldSlotEid === null) continue;

            const options = resetOptions(mutatedVehicleOptions);
            options.playerId = PlayerRef.id[vehicleEid];
            options.teamId = TeamRef.id[vehicleEid];
            options.x = RigidBodyState.position.get(vehicleEid, 0);
            options.y = RigidBodyState.position.get(vehicleEid, 1);
            options.rotation = RigidBodyState.rotation[vehicleEid];

            fillSlot(emptyShieldSlotEid, options);

            regenTimers.set(turretEid, currentTime);
        }

        for (const turretEid of regenTimers.keys()) {
            if (!turretEids.includes(turretEid)) {
                regenTimers.delete(turretEid);
            }
        }
    };
}
