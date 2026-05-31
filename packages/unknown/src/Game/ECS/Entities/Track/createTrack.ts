import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { PhysicalWorld } from '../../../Physical/initPhysicalWorld.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { TrackSide } from '../../Components/Track.ts';
import { VehicleOptions } from '../Vehicle/Options.ts';
import { spawnRectangleCarrier, SpawnCtx } from '../spawnPart.ts';
import { getPhysicsWorldComponents, PhysicsWorld } from '../../createPhysicsWorld.ts';
import { getRenderWorldComponents } from '../../createRenderWorld.ts';
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
    world: PhysicsWorld,
    physicalWorld: PhysicalWorld,
    options: TrackOptions,
    vehicleRenderEid: number,
    vehiclePid: number,
): [number, number, number] {
    const { Track, Joint } = getPhysicsWorldComponents(world);
    const renderWorld = Worlds.renderWorld;
    const { Parent, Children } = getRenderWorldComponents(renderWorld);

    options.belongsCollisionGroup = CollisionGroup.NONE;
    options.interactsCollisionGroup = CollisionGroup.NONE;
    options.belongsSolverGroup = CollisionGroup.NONE;
    options.interactsSolverGroup = CollisionGroup.NONE;

    const ctx: SpawnCtx = { physicsWorld: world, renderWorld, physicalWorld };
    const [trackPhysEid, trackRenderEid, trackPid] = spawnRectangleCarrier(ctx, options);

    Track.addComponent(world, trackPhysEid, options.trackSide, options.trackLength);

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
    Joint.addComponent(world, trackPhysEid, joint.handle);

    Parent.addComponent(renderWorld, trackRenderEid, vehicleRenderEid);
    Children.addComponent(renderWorld, trackRenderEid);
    Children.addChildren(vehicleRenderEid, trackRenderEid);

    return [trackPhysEid, trackRenderEid, trackPid];
}
