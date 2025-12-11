import { hasComponent, query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { Tank } from '../../Components/Tank.ts';
import { TankPart } from '../../Components/TankPart.ts';
import { RigidBodyState } from '../../Components/Physical.ts';
import { Vector2 } from '@dimforge/rapier2d-simd';
import { abs, cos, PI, sign, sin } from '../../../../../../../lib/math.ts';
import { Slot } from '../../Components/Slot.ts';
import { SlotPartType } from '../../Components/SlotConfig.ts';
import { Children } from '../../Components/Children.ts';

export function createVisualizationTracksSystem({ world, physicalWorld } = GameDI) {
    const vect2 = new Vector2(0, 0);

    return (_delta: number) => {
        const tankEids = query(world, [Tank]);

        for (const tankEid of tankEids) {
            const caterpillarsLimit = Tank.caterpillarsLength[tankEid] / 2;
            const linvel = RigidBodyState.linvel.getBatch(tankEid);
            const angvel = RigidBodyState.angvel[tankEid];
            const rotation = RigidBodyState.rotation[tankEid];

            const forwardX = cos(rotation - PI / 2);
            const forwardY = sin(rotation - PI / 2);
            const speed = linvel[0] * forwardX + linvel[1] * forwardY;

            // Iterate children of tank, filter by Slot component
            const childCount = Children.entitiesCount[tankEid];
            for (let i = 0; i < childCount; i++) {
                const slotEid = Children.entitiesIds.get(tankEid, i);
                
                // Skip non-slot children
                if (!hasComponent(world, slotEid, Slot)) continue;
                
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
                const fillerEid = Slot.getFillerEid(slotEid);
                if (fillerEid === 0) continue;

                const jointPid = TankPart.jointPid[fillerEid];
                const joint = physicalWorld.getImpulseJoint(jointPid);

                if (jointPid === 0 || joint == null) continue;

                vect2.x = anchorX;
                vect2.y = anchorY;
                joint.setAnchor1(vect2);
            }
        }
    };
}