import { query } from 'bitecs';
import { Worlds } from '../../DI/Worlds.ts';
import { VehicleType } from '../Components/Vehicle.ts';
import { SlotPartType } from '../Components/SlotConfig.ts';
import { fillSlot } from '../Entities/Vehicle/VehicleParts.ts';
import { mutatedVehicleOptions, resetOptions } from '../Entities/Vehicle/Options.ts';
import { isSlotEmpty } from '../Utils/SlotUtils.ts';
import { ShieldConfig } from '../../Config/index.ts';
import { getPhysicsWorldComponents } from '../createPhysicsWorld.ts';
import { getBrainWorldComponents } from '../createBrainWorld.ts';
import { getSlotWorldComponents, SlotWorld } from '../createSlotWorld.ts';
import { getNodeParent, getNodePhysics, getNodeSlots } from '../refs.ts';

const SHIELD_REGEN_INTERVAL = ShieldConfig.regenInterval;

export function createShieldRegenerationSystem({ physicsWorld, slotWorld, brainWorld } = Worlds) {
    const { RigidBodyState } = getPhysicsWorldComponents(physicsWorld);
    const { Vehicle, TurretController, PlayerRef, TeamRef } = getBrainWorldComponents(brainWorld);
    const regenTimers = new Map<number, number>();

    function findFirstEmptyShieldSlot(slotWorld: SlotWorld, turretNode: number): number | null {
        const { Slot } = getSlotWorldComponents(slotWorld);

        for (const slotEid of getNodeSlots(turretNode)) {
            if (Slot.partType[slotEid] !== SlotPartType.Shield) continue;
            if (!isSlotEmpty(slotWorld, slotEid)) continue;

            return slotEid;
        }

        return null;
    }

    return (_delta: number) => {
        const currentTime = performance.now();

        // Node-rooted: iterate turret NODES (TurretController) — same set as the old
        // query([VehicleTurret]), since every turret carries a TurretController brain.
        const turretNodes = query(brainWorld, [TurretController]);

        for (let i = 0; i < turretNodes.length; i++) {
            const turretNode = turretNodes[i];
            // turret node -> Brain parent (hull node = hull-brain) -> hull physics (vehicle atom).
            const hullBrain = getNodeParent(turretNode);
            const vehicleEid = getNodePhysics(hullBrain);

            if (Vehicle.type[hullBrain] !== VehicleType.Harvester) continue;

            const lastRegenTime = regenTimers.get(turretNode) ?? 0;
            if (currentTime - lastRegenTime < SHIELD_REGEN_INTERVAL) continue;

            const emptyShieldSlotEid = findFirstEmptyShieldSlot(slotWorld, turretNode);
            if (emptyShieldSlotEid === null) continue;

            const options = resetOptions(mutatedVehicleOptions);
            options.playerId = PlayerRef.id[hullBrain];
            options.teamId = TeamRef.id[hullBrain];
            options.x = RigidBodyState.position.get(vehicleEid, 0);
            options.y = RigidBodyState.position.get(vehicleEid, 1);
            options.rotation = RigidBodyState.rotation[vehicleEid];

            // The shield slot is owned by the turret node (its carrier).
            fillSlot(emptyShieldSlotEid, turretNode, options);

            regenTimers.set(turretNode, currentTime);
        }

        for (const turretNode of regenTimers.keys()) {
            if (!turretNodes.includes(turretNode)) {
                regenTimers.delete(turretNode);
            }
        }
    };
}
