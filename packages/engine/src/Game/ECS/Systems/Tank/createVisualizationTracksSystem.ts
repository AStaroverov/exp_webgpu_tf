import { query } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { TrackSide } from "../../Components/Track.ts";
import { Vector2 } from "@dimforge/rapier2d-simd";
import { abs, cos, sign, sin } from "../../../../../../../lib/math.ts";
import { getSlotFillerEid, isSlot } from "../../Utils/SlotUtils.ts";
import { getGameComponents } from "../../createGameWorld.ts";

const TURN_FACTOR = 0.7;
const ANGULAR_SCALE = 50;

export function createVisualizationTracksSystem({ world, physicalWorld } = GameDI) {
  const { Track, RigidBodyState, Slot, Children, CompoundPart } = getGameComponents(world);
  const vect2 = new Vector2(0, 0);

  return (_delta: number) => {
    const trackEids = query(world, [Track]);

    for (const trackEid of trackEids) {
      const trackLimit = Track.length.get(trackEid) / 2;
      const trackSide = Track.side.get(trackEid);

      const linvel = RigidBodyState.linvel.getBatch(trackEid);
      const angvel = RigidBodyState.angvel[trackEid];
      const rotation = RigidBodyState.rotation[trackEid];

      const forwardX = cos(rotation);
      const forwardY = -sin(rotation);
      const forwardSpeed = linvel[0] * forwardX + linvel[1] * forwardY;

      const rotationContribution = angvel * ANGULAR_SCALE * TURN_FACTOR;
      const trackRotationDelta =
        trackSide === TrackSide.Left ? rotationContribution : -rotationContribution;

      const speed = forwardSpeed + trackRotationDelta;

      let delta = speed / 100;
      delta -= delta % 0.01;

      if (abs(delta) < 0.05) continue;

      const childCount = Children.entitiesCount.get(trackEid);
      for (let i = 0; i < childCount; i++) {
        const slotEid = Children.entitiesIds.get(trackEid, i);

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

        const fillerEid = getSlotFillerEid(slotEid);
        if (fillerEid === 0) continue;

        vect2.x = anchorX;
        vect2.y = anchorY;

        CompoundPart.anchorX.set(fillerEid, anchorX);
        CompoundPart.anchorY.set(fillerEid, anchorY);

        physicalWorld
          .getCollider(CompoundPart.colliderHandle.get(fillerEid))
          ?.setTranslationWrtParent(vect2);
      }
    }
  };
}
