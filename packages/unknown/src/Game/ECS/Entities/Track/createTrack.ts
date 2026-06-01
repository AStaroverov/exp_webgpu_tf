import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { addEntity } from 'bitecs';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { TrackSide } from '../../Components/Track.ts';
import { VehicleOptions } from '../Vehicle/Options.ts';
import { spawnRectangleCarrier } from '../spawnPart.ts';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { getHullBrainByRender } from '../Vehicle/VehicleBase.ts';
import { setNodeRender, linkBrainChild } from '../../refs.ts';
import { Worlds } from '../../../DI/Worlds.ts';

export type TrackOptions = VehicleOptions & {
    trackSide: TrackSide;
    trackLength: number;
    anchorX: number;
    anchorY: number;
};

const jointParentAnchor = new Vector2(0, 0);
const jointChildAnchor = new Vector2(0, 0);

// Returns [trackPhysEid, trackRenderEid, trackPid]
export function createTrack(
    options: TrackOptions,
    vehicleRenderEid: number,
    vehiclePid: number,
    { physicsWorld, physicalWorld, brainWorld } = Worlds,
): [number, number, number] {
    const { Track, Joint } = getPhysicsWorldComponents(physicsWorld);

    options.belongsCollisionGroup = CollisionGroup.NONE;
    options.interactsCollisionGroup = CollisionGroup.NONE;
    options.belongsSolverGroup = CollisionGroup.NONE;
    options.interactsSolverGroup = CollisionGroup.NONE;

    // Brain-first: the track node before its physics/render presentation.
    const trackNode = addEntity(brainWorld);

    const [trackPhysEid, trackRenderEid, trackPid] = spawnRectangleCarrier(options);

    Track.addComponent(physicsWorld, trackPhysEid, options.trackSide, options.trackLength);

    // Node->render presentation + Brain hierarchy: track node is a child of the hull
    // node; it reaches its track atom downward via getNodePhysics(trackNode).
    const hullBrain = getHullBrainByRender(vehicleRenderEid);
    setNodeRender(trackNode, trackRenderEid);
    linkBrainChild(hullBrain, trackNode);

    jointParentAnchor.x = options.anchorX;
    jointParentAnchor.y = options.anchorY;
    jointChildAnchor.x = 0;
    jointChildAnchor.y = 0;

    const vehicleBody = physicalWorld.getRigidBody(vehiclePid);
    const trackBody = physicalWorld.getRigidBody(trackPid);

    const joint = physicalWorld.createImpulseJoint(
        JointData.fixed(jointParentAnchor, 0, jointChildAnchor, 0),
        vehicleBody,
        trackBody,
        false,
    );
    Joint.addComponent(physicsWorld, trackPhysEid, joint.handle);

    // No render parent/children: the track carrier is physics-driven (own world
    // transform from mirrorSync) → render ROOT, not a child (parenting double-composes).

    return [trackPhysEid, trackRenderEid, trackPid];
}
