import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { addTransformComponents } from '../../../../../../renderer/src/ECS/Components/Transform.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { Children } from '../../Components/Children.ts';
import { Parent } from '../../Components/Parent.ts';
import { createRectangleRigidGroup } from '../../Components/RigidGroup.ts';
import { Wheel, WheelDrive, WheelPosition, WheelSteerable } from '../../Components/Wheel.ts';
import { Joint } from '../../Components/Joint.ts';
import { JointMotor } from '../../Components/JointMotor.ts';
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

/**
 * Creates a wheel entity with its own rigid body, attached to the vehicle via revolute joint.
 * The wheel can optionally be steerable (for front wheels) and/or driven (receives power).
 * 
 * @param options - Wheel creation options
 * @param vehicleEid - Parent vehicle entity ID
 * @param vehiclePid - Parent vehicle physics ID
 * @returns [wheelEid, wheelPid] - Wheel entity ID and physics ID
 */
export function createWheel(
    options: WheelOptions,
    vehicleEid: number,
    vehiclePid: number,
    { world, physicalWorld } = GameDI,
): [number, number] {
    // Wheels have minimal collision
    options.belongsCollisionGroup = CollisionGroup.NONE;
    options.interactsCollisionGroup = CollisionGroup.NONE;
    options.belongsSolverGroup = CollisionGroup.NONE;
    options.interactsSolverGroup = CollisionGroup.NONE;

    const [wheelEid, wheelPid] = createRectangleRigidGroup(options);
    
    // Add Wheel component
    Wheel.addComponent(world, wheelEid);

    // Create revolute joint to attach wheel to vehicle (allows rotation for steering)
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
    Joint.addComponent(world, wheelEid, joint.handle);

    // Add steerable component if this is a steerable wheel
    if (options.isSteerable) {
        WheelSteerable.addComponent(
            world,
            wheelEid,
            options.maxSteeringAngle ?? PI / 6, // Default ~30 degrees
            options.steeringSpeed ?? PI * 2,
        );
        JointMotor.addComponent(world, wheelEid);
    }

    // Add drive component if this wheel receives power
    if (options.isDrive) {
        WheelDrive.addComponent(world, wheelEid);
    }

    // Setup entity hierarchy
    addTransformComponents(world, wheelEid);
    Parent.addComponent(world, wheelEid, vehicleEid);
    Children.addComponent(world, wheelEid);
    Children.addChildren(vehicleEid, wheelEid);

    return [wheelEid, wheelPid];
}

