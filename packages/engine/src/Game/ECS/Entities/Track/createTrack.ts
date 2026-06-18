import { JointData, Vector2 } from "@dimforge/rapier2d-simd";
import { addTransformComponents } from "renderer/src/ECS/Components/Transform.ts";
import { GameDI } from "../../../DI/GameDI.ts";
import { CollisionGroup } from "../../../Physical/createRigid.ts";
import { TrackSide } from "../../Components/Track.ts";
import { VehicleOptions } from "../Vehicle/Options.ts";
import { createRectangleRigidGroup } from "../../Components/RigidGroup.ts";
import { getGameComponents } from "../../createGameWorld.ts";

export type TrackOptions = VehicleOptions & {
  trackSide: TrackSide;
  trackLength: number;
  anchorX: number;
  anchorY: number;
};

const jointParentAnchor = new Vector2(0, 0);
const jointChildAnchor = new Vector2(0, 0);

export function createTrack(
  options: TrackOptions,
  vehicleEid: number,
  vehiclePid: number,
  { world, physicalWorld } = GameDI,
): [number, number] {
  const { Track, Joint, Parent, Children } = getGameComponents(world);

  options.belongsCollisionGroup = CollisionGroup.NONE;
  options.interactsCollisionGroup = CollisionGroup.NONE;
  options.belongsSolverGroup = CollisionGroup.NONE;
  options.interactsSolverGroup = CollisionGroup.NONE;

  const [trackEid, trackPid] = createRectangleRigidGroup(options);

  Track.addComponent(world, trackEid, options.trackSide, options.trackLength);

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

  addTransformComponents(world, trackEid);
  Parent.addComponent(world, trackEid, vehicleEid);
  Children.addComponent(world, trackEid);
  Children.addChildren(vehicleEid, trackEid);

  return [trackEid, trackPid];
}
