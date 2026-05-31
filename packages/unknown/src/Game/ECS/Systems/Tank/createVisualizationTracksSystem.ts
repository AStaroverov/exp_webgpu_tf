import { query } from 'bitecs';
import { TrackSide } from '../../Components/Track.ts';
import { Vector2 } from '@dimforge/rapier2d-simd';
import { abs, cos, sign, sin } from '../../../../../../../lib/math.ts';
import { getSlotFillerEid, isSlot } from '../../Utils/SlotUtils.ts';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { getRenderWorldComponents } from '../../createRenderWorld.ts';
import { BridgeDI } from '../../../DI/BridgeDI.ts';
import { Worlds } from '../../../DI/Worlds.ts';

const TURN_FACTOR = 0.7;
const ANGULAR_SCALE = 50;

export function createVisualizationTracksSystem({ physicsWorld, renderWorld, physicalWorld } = Worlds) {
    const { Track, Joint, RigidBodyState } = getPhysicsWorldComponents(physicsWorld);
    const vect2 = new Vector2(0, 0);

    return (_delta: number) => {
        const { Slot, Children } = getRenderWorldComponents(renderWorld);
        const trackEids = query(physicsWorld, [Track]);

        for (const trackEid of trackEids) {
            const trackLimit = Track.length[trackEid] / 2;
            const trackSide = Track.side[trackEid];

            const linvel = RigidBodyState.linvel.getBatch(trackEid);
            const angvel = RigidBodyState.angvel[trackEid];
            const rotation = RigidBodyState.rotation[trackEid];

            const forwardX = cos(rotation);
            const forwardY = -sin(rotation);
            const forwardSpeed = linvel[0] * forwardX + linvel[1] * forwardY;

            const rotationContribution = angvel * ANGULAR_SCALE * TURN_FACTOR;
            const trackRotationDelta = trackSide === TrackSide.Left
                ? rotationContribution
                : -rotationContribution;

            const speed = forwardSpeed + trackRotationDelta;

            let delta = speed / 100;
            delta -= delta % 0.01;

            if (abs(delta) < 0.05) continue;

            const trackRenderEid = BridgeDI.getRenderOf(trackEid);
            const childCount = Children.entitiesCount[trackRenderEid];
            for (let i = 0; i < childCount; i++) {
                const slotEid = Children.entitiesIds.get(trackRenderEid, i);

                if (!isSlot(renderWorld, slotEid)) continue;

                let anchorX = Slot.anchorX[slotEid];
                let anchorY = Slot.anchorY[slotEid];

                anchorX += delta;
                anchorX -= anchorX % 0.01;

                if (abs(anchorX) > trackLimit) {
                    anchorX = -sign(anchorX) * (trackLimit + (trackLimit - abs(anchorX)));
                }

                Slot.anchorX[slotEid] = anchorX;
                Slot.anchorY[slotEid] = anchorY;

                const fillerRenderEid = getSlotFillerEid(renderWorld, slotEid);
                if (fillerRenderEid === 0) continue;
                const fillerEid = BridgeDI.getPhysicsOf(fillerRenderEid);

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
