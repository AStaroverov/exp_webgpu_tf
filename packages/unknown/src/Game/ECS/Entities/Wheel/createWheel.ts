import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { addEntity } from 'bitecs';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { spawnRectangleCarrier } from '../spawnPart.ts';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { getHullBrainByRender } from '../Vehicle/VehicleBase.ts';
import { setNodeRender, linkBrainChild } from '../../refs.ts';
import { Worlds } from '../../../DI/Worlds.ts';
import { WheelPosition } from '../../Components/Wheel.ts';
import { VehicleOptions } from '../Vehicle/Options.ts';
import { PI } from '../../../../../../../lib/math.ts';

export type WheelOptions = VehicleOptions & {
    wheelPosition: WheelPosition;
    anchorX: number;
    anchorY: number;
    isSteerable?: boolean;
    isDrive?: boolean;
    maxSteeringAngle?: number;
    steeringSpeed?: number;
};

const jointParentAnchor = new Vector2(0, 0);
const jointChildAnchor = new Vector2(0, 0);

// Returns [wheelPhysEid, wheelRenderEid, wheelPid]
export function createWheel(
    options: WheelOptions,
    vehicleRenderEid: number,
    vehiclePid: number,
    { physicsWorld, physicalWorld, brainWorld } = Worlds,
): [number, number, number] {
    const { Wheel, WheelSteerable, WheelDrive, Joint, JointMotor } = getPhysicsWorldComponents(physicsWorld);

    options.belongsCollisionGroup = CollisionGroup.NONE;
    options.interactsCollisionGroup = CollisionGroup.NONE;
    options.belongsSolverGroup = CollisionGroup.NONE;
    options.interactsSolverGroup = CollisionGroup.NONE;

    // Brain-first: the wheel node before its physics/render presentation.
    const wheelNode = addEntity(brainWorld);

    const [wheelPhysEid, wheelRenderEid, wheelPid] = spawnRectangleCarrier(options);

    Wheel.addComponent(physicsWorld, wheelPhysEid);

    // Node->render presentation + Brain hierarchy: wheel node is a child of the hull
    // node; it reaches its wheel atom downward via getNodePhysics(wheelNode).
    const hullBrain = getHullBrainByRender(vehicleRenderEid);
    setNodeRender(wheelNode, wheelRenderEid);
    linkBrainChild(hullBrain, wheelNode);

    jointParentAnchor.x = options.anchorX;
    jointParentAnchor.y = options.anchorY;
    jointChildAnchor.x = 0;
    jointChildAnchor.y = 0;

    const vehicleBody = physicalWorld.getRigidBody(vehiclePid);
    const wheelBody = physicalWorld.getRigidBody(wheelPid);

    const joint = physicalWorld.createImpulseJoint(
        options.isSteerable
            ? JointData.revolute(jointParentAnchor, jointChildAnchor)
            : JointData.fixed(jointParentAnchor, 0, jointChildAnchor, 0),
        vehicleBody,
        wheelBody,
        false,
    );
    Joint.addComponent(physicsWorld, wheelPhysEid, joint.handle);

    if (options.isSteerable) {
        WheelSteerable.addComponent(
            physicsWorld,
            wheelPhysEid,
            options.maxSteeringAngle ?? PI / 6,
            options.steeringSpeed ?? PI * 2,
        );
        JointMotor.addComponent(physicsWorld, wheelPhysEid);
    }

    if (options.isDrive) {
        WheelDrive.addComponent(physicsWorld, wheelPhysEid);
    }

    // No render parent/children: the wheel carrier is physics-driven (own world
    // transform from mirrorSync) → render ROOT, not a child (parenting double-composes).

    return [wheelPhysEid, wheelRenderEid, wheelPid];
}
