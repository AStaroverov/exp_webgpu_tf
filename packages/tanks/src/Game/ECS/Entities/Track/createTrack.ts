import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { addTransformComponents } from '../../../../../../renderer/src/ECS/Components/Transform.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { Children } from '../../Components/Children.ts';
import { Parent } from '../../Components/Parent.ts';
import { Track, TrackSide } from '../../Components/Track.ts';
import { Joint } from '../../Components/Joint.ts';
import { VehicleOptions } from '../Vehicle/Options.ts';
import { createRectangleRigidGroup } from '../../Components/RigidGroup.ts';

export type TrackOptions = VehicleOptions & {
    trackSide: TrackSide;
    trackLength: number;
    anchorX: number;
    anchorY: number;
};

const jointParentAnchor = new Vector2(0, 0);
const jointChildAnchor = new Vector2(0, 0);

/**
 * Creates a track entity with its own rigid body, attached to the vehicle via fixed joint.
 * The track acts as an independent drive unit.
 * 
 * @param options - Track creation options
 * @param vehicleEid - Parent vehicle entity ID
 * @param vehiclePid - Parent vehicle physics ID
 * @returns [trackEid, trackPid] - Track entity ID and physics ID
 */
export function createTrack(
    options: TrackOptions,
    vehicleEid: number,
    vehiclePid: number,
    { world, physicalWorld } = GameDI,
): [number, number] {
    // Track has minimal collision - mainly for structural purposes
    options.belongsCollisionGroup = CollisionGroup.NONE;
    options.interactsCollisionGroup = CollisionGroup.NONE;
    options.belongsSolverGroup = CollisionGroup.NONE;
    options.interactsSolverGroup = CollisionGroup.NONE;

    // const [trackEid, trackPid] = createRectangleRigidGroup(options);
    const [trackEid, trackPid] = createRectangleRigidGroup(options);
    
    // Add Track component
    Track.addComponent(world, trackEid, options.trackSide, options.trackLength);

    // Create fixed joint to attach track to vehicle
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
    Joint.addComponent(world, trackEid, joint.handle);

    // Setup entity hierarchy
    addTransformComponents(world, trackEid);
    Parent.addComponent(world, trackEid, vehicleEid);
    Children.addComponent(world, trackEid);
    Children.addChildren(vehicleEid, trackEid);

    return [trackEid, trackPid];
}

