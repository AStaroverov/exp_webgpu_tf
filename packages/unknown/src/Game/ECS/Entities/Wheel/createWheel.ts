import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { PhysicalWorld } from '../../../Physical/initPhysicalWorld.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { spawnRectangleCarrier, SpawnCtx } from '../spawnPart.ts';
import { getPhysicsWorldComponents, PhysicsWorld } from '../../createPhysicsWorld.ts';
import { getRenderWorldComponents } from '../../createRenderWorld.ts';
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
    world: PhysicsWorld,
    physicalWorld: PhysicalWorld,
    options: WheelOptions,
    vehicleRenderEid: number,
    vehiclePid: number,
): [number, number, number] {
    const { Wheel, WheelSteerable, WheelDrive, Joint, JointMotor } = getPhysicsWorldComponents(world);
    const renderWorld = Worlds.renderWorld;
    const { Parent, Children } = getRenderWorldComponents(renderWorld);

    options.belongsCollisionGroup = CollisionGroup.NONE;
    options.interactsCollisionGroup = CollisionGroup.NONE;
    options.belongsSolverGroup = CollisionGroup.NONE;
    options.interactsSolverGroup = CollisionGroup.NONE;

    const ctx: SpawnCtx = { physicsWorld: world, renderWorld, physicalWorld };
    const [wheelPhysEid, wheelRenderEid, wheelPid] = spawnRectangleCarrier(ctx, options);

    Wheel.addComponent(world, wheelPhysEid);

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
    Joint.addComponent(world, wheelPhysEid, joint.handle);

    if (options.isSteerable) {
        WheelSteerable.addComponent(
            world,
            wheelPhysEid,
            options.maxSteeringAngle ?? PI / 6,
            options.steeringSpeed ?? PI * 2,
        );
        JointMotor.addComponent(world, wheelPhysEid);
    }

    if (options.isDrive) {
        WheelDrive.addComponent(world, wheelPhysEid);
    }

    Parent.addComponent(renderWorld, wheelRenderEid, vehicleRenderEid);
    Children.addComponent(renderWorld, wheelRenderEid);
    Children.addChildren(vehicleRenderEid, wheelRenderEid);

    return [wheelPhysEid, wheelRenderEid, wheelPid];
}
