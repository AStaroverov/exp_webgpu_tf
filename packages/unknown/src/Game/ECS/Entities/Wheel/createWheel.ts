import { JointData, Vector2 } from "@dimforge/rapier2d-simd";
import { addTransformComponents } from "../../../../../../renderer/src/ECS/Components/Transform.ts";
import { GameDI } from "../../../DI/GameDI.ts";
import { CollisionGroup } from "../../../Physical/createRigid.ts";
import { createRectangleRigidGroup } from "../../Components/RigidGroup.ts";
import { WheelPosition } from "../../Components/Wheel.ts";
import { VehicleOptions } from "../Vehicle/Options.ts";
import { PI } from "../../../../../../../lib/math.ts";
import { getGameComponents } from "../../createGameWorld.ts";

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

export function createWheel(
  options: WheelOptions,
  vehicleEid: number,
  vehiclePid: number,
  { world, physicalWorld } = GameDI,
): [number, number] {
  const { Wheel, WheelSteerable, WheelDrive, Joint, JointMotor, Parent, Children } =
    getGameComponents(world);

  options.belongsCollisionGroup = CollisionGroup.NONE;
  options.interactsCollisionGroup = CollisionGroup.NONE;
  options.belongsSolverGroup = CollisionGroup.NONE;
  options.interactsSolverGroup = CollisionGroup.NONE;

  const [wheelEid, wheelPid] = createRectangleRigidGroup(options);

  Wheel.addComponent(world, wheelEid);

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

  if (options.isSteerable) {
    WheelSteerable.addComponent(
      world,
      wheelEid,
      options.maxSteeringAngle ?? PI / 6,
      options.steeringSpeed ?? PI * 2,
    );
    JointMotor.addComponent(world, wheelEid);
  }

  if (options.isDrive) {
    WheelDrive.addComponent(world, wheelEid);
  }

  addTransformComponents(world, wheelEid);
  Parent.addComponent(world, wheelEid, vehicleEid);
  Children.addComponent(world, wheelEid);
  Children.addChildren(vehicleEid, wheelEid);

  return [wheelEid, wheelPid];
}
