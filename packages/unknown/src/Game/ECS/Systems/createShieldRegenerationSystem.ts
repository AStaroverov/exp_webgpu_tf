import { query } from 'bitecs';
import { Worlds } from '../../DI/Worlds.ts';
import { VehicleType } from '../Components/Vehicle.ts';
import { SlotPartType } from '../Components/SlotConfig.ts';
import { fillSlot } from '../Entities/Vehicle/VehicleParts.ts';
import { mutatedVehicleOptions, resetOptions } from '../Entities/Vehicle/Options.ts';
import { isSlot, isSlotEmpty } from '../Utils/SlotUtils.ts';
import { ShieldConfig } from '../../Config/index.ts';
import { getPhysicsWorldComponents } from '../createPhysicsWorld.ts';
import { getRenderWorldComponents, RenderGameWorld } from '../createRenderWorld.ts';
import { BridgeDI } from '../../DI/BridgeDI.ts';

const SHIELD_REGEN_INTERVAL = ShieldConfig.regenInterval;

export function createShieldRegenerationSystem({ physicsWorld, renderWorld, physicalWorld } = Worlds) {
    const { Vehicle, VehicleTurret, RigidBodyState, PlayerRef, TeamRef } = getPhysicsWorldComponents(physicsWorld);
    const regenTimers = new Map<number, number>();

    function findFirstEmptyShieldSlot(renderWorld: RenderGameWorld, parentRenderEid: number): number | null {
        const { Slot, Children } = getRenderWorldComponents(renderWorld);
        const childCount = Children.entitiesCount[parentRenderEid];

        for (let i = 0; i < childCount; i++) {
            const childEid = Children.entitiesIds.get(parentRenderEid, i);

            if (!isSlot(renderWorld, childEid)) continue;
            if (Slot.partType[childEid] !== SlotPartType.Shield) continue;
            if (!isSlotEmpty(renderWorld, childEid)) continue;

            return childEid;
        }

        return null;
    }

    return (_delta: number) => {
        const currentTime = performance.now();

        const turretEids = query(physicsWorld, [VehicleTurret]);

        for (let i = 0; i < turretEids.length; i++) {
            const turretEid = turretEids[i];
            // turret phys -> turret render -> vehicle render -> vehicle phys
            const turretRenderEid = BridgeDI.getRenderOf(turretEid);
            const { Parent } = getRenderWorldComponents(renderWorld);
            const vehicleEid = BridgeDI.getPhysicsOf(Parent.id[turretRenderEid]);

            if (Vehicle.type[vehicleEid] !== VehicleType.Harvester) continue;

            const lastRegenTime = regenTimers.get(turretEid) ?? 0;
            if (currentTime - lastRegenTime < SHIELD_REGEN_INTERVAL) continue;

            const emptyShieldSlotEid = findFirstEmptyShieldSlot(renderWorld, turretRenderEid);
            if (emptyShieldSlotEid === null) continue;

            const options = resetOptions(mutatedVehicleOptions);
            options.playerId = PlayerRef.id[vehicleEid];
            options.teamId = TeamRef.id[vehicleEid];
            options.x = RigidBodyState.position.get(vehicleEid, 0);
            options.y = RigidBodyState.position.get(vehicleEid, 1);
            options.rotation = RigidBodyState.rotation[vehicleEid];

            fillSlot(renderWorld, physicalWorld, emptyShieldSlotEid, options);

            regenTimers.set(turretEid, currentTime);
        }

        for (const turretEid of regenTimers.keys()) {
            if (!turretEids.includes(turretEid)) {
                regenTimers.delete(turretEid);
            }
        }
    };
}
