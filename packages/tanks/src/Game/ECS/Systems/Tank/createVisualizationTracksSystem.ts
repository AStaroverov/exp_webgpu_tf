import { query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { Tank } from '../../Components/Tank.ts';
import { VehiclePart } from '../../Components/VehiclePart.ts';
import { RigidBodyState } from '../../Components/Physical.ts';
import { Vector2 } from '@dimforge/rapier2d-simd';
import { abs, cos, PI, sign, sin } from '../../../../../../../lib/math.ts';
import { Slot } from '../../Components/Slot.ts';
import { SlotPartType } from '../../Components/SlotConfig.ts';
import { Children } from '../../Components/Children.ts';
import { getSlotFillerEid, isSlot } from '../../Utils/SlotUtils.ts';

export function createVisualizationTracksSystem({ world, physicalWorld } = GameDI) {
    const vect2 = new Vector2(0, 0);

    return (_delta: number) => {
        const vehicleEids = query(world, [Tank]);

        for (const vehicleEid of vehicleEids) {
            const caterpillarsLimit = Tank.caterpillarsLength[vehicleEid] / 2;
            const linvel = RigidBodyState.linvel.getBatch(vehicleEid);
            const angvel = RigidBodyState.angvel[vehicleEid];
            const rotation = RigidBodyState.rotation[vehicleEid];

            const forwardX = cos(rotation - PI / 2);
            const forwardY = sin(rotation - PI / 2);
            const speed = linvel[0] * forwardX + linvel[1] * forwardY;

            // Iterate children of vehicle, filter by Slot component
            const childCount = Children.entitiesCount[vehicleEid];
            for (let i = 0; i < childCount; i++) {
                const slotEid = Children.entitiesIds.get(vehicleEid, i);
                
                // Skip non-slot children
                if (!isSlot(slotEid)) continue;
                
                // Skip non-caterpillar slots by partType
                if (Slot.partType[slotEid] !== SlotPartType.Caterpillar) continue;
                
                let anchorX = Slot.anchorX[slotEid];
                let anchorY = Slot.anchorY[slotEid];
                
                const angFactor = anchorX > 0 ? -0.8 : 0.8;
                let delta = (speed / 100 + (angvel * angFactor));
                delta -= delta % 0.01;

                if (abs(delta) < 0.05) continue;

                anchorY -= delta;
                anchorY -= anchorY % 0.01;

                if (abs(anchorY) > caterpillarsLimit) {
                    anchorY = -sign(anchorY) * (caterpillarsLimit + (caterpillarsLimit - abs(anchorY)));
                }

                Slot.anchorX[slotEid] = anchorX;
                Slot.anchorY[slotEid] = anchorY;

                // Get filler from slot's children (O(1))
                const fillerEid = getSlotFillerEid(slotEid);
                if (fillerEid === 0) continue;

                const jointPid = VehiclePart.jointPid[fillerEid];
                const joint = physicalWorld.getImpulseJoint(jointPid);

                if (jointPid === 0 || joint == null) continue;

                vect2.x = anchorX;
                vect2.y = anchorY;
                joint.setAnchor1(vect2);
            }
        }
    };
}