import { query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { Track, TrackSide } from '../../Components/Track.ts';
import { Joint } from '../../Components/Joint.ts';
import { RigidBodyState } from '../../Components/Physical.ts';
import { Vector2 } from '@dimforge/rapier2d-simd';
import { abs, cos, sign, sin } from '../../../../../../../lib/math.ts';
import { Slot } from '../../Components/Slot.ts';
import { Children } from '../../Components/Children.ts';
import { getSlotFillerEid, isSlot } from '../../Utils/SlotUtils.ts';

// How much angular velocity affects track differential (same as in TrackControlSystem)
const TURN_FACTOR = 0.7;
// Scale for angular velocity contribution
const ANGULAR_SCALE = 50;

export function createVisualizationTracksSystem({ world, physicalWorld } = GameDI) {
    const vect2 = new Vector2(0, 0);

    return (_delta: number) => {
        const trackEids = query(world, [Track]);

        for (const trackEid of trackEids) {
            const trackLimit = Track.length[trackEid] / 2;
            const trackSide = Track.side[trackEid];
            
            // Get velocities
            const linvel = RigidBodyState.linvel.getBatch(trackEid);
            const angvel = RigidBodyState.angvel[trackEid];
            const rotation = RigidBodyState.rotation[trackEid];
            
            // Calculate linear speed in vehicle's forward direction
            const forwardX = cos(rotation);
            const forwardY = -sin(rotation);
            const forwardSpeed = linvel[0] * forwardX + linvel[1] * forwardY;
            
            // Calculate rotation contribution based on track side
            // Left track: positive angvel (turning left) -> slower, negative angvel -> faster
            // Right track: positive angvel (turning left) -> faster, negative angvel -> slower
            const rotationContribution = angvel * ANGULAR_SCALE * TURN_FACTOR;
            const trackRotationDelta = trackSide === TrackSide.Left 
                ? rotationContribution 
                : -rotationContribution;
            
            const speed = forwardSpeed + trackRotationDelta;

            let delta = speed / 100;
            delta -= delta % 0.01;

            if (abs(delta) < 0.05) continue;

            // Iterate children of track, filter by Slot component
            const childCount = Children.entitiesCount[trackEid];
            for (let i = 0; i < childCount; i++) {
                const slotEid = Children.entitiesIds.get(trackEid, i);
                
                // Skip non-slot children
                if (!isSlot(slotEid)) continue;
                
                let anchorX = Slot.anchorX[slotEid];
                let anchorY = Slot.anchorY[slotEid];

                anchorX += delta;
                anchorX -= anchorX % 0.01;

                if (abs(anchorX) > trackLimit) {
                    anchorX = -sign(anchorX) * (trackLimit + (trackLimit - abs(anchorX)));
                }

                Slot.anchorX[slotEid] = anchorX;
                Slot.anchorY[slotEid] = anchorY;

                // Get filler from slot's children
                const fillerEid = getSlotFillerEid(slotEid);
                if (fillerEid === 0) continue;

                const jointPid = Joint.pid[fillerEid];
                const joint = physicalWorld.getImpulseJoint(jointPid);

                if (jointPid === 0 || joint == null) continue;

                vect2.x = anchorX;
                vect2.y = anchorY;
                joint.setAnchor1(vect2);
            }
        }
    };
}