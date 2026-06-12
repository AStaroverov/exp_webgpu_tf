import { query } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { getGameComponents } from "../createGameWorld.ts";

export function createRigidBodyStateSystem({ world, physicalWorld } = GameDI) {
  const { RigidBodyRef, RigidBodyState } = getGameComponents(world);
  const rawBodies = physicalWorld.bodies.raw;

  return () => {
    const entities = query(world, [RigidBodyRef, RigidBodyState]);

    for (let i = 0; i < entities.length; i++) {
      const eid = entities[i];
      const pid = RigidBodyRef.id[eid];
      // A sleeping body hasn't moved since the last sync. Bodies are created
      // awake and fixed bodies never sleep, so every body gets a first sync.
      if (rawBodies.rbIsSleeping(pid)) continue;

      const translation = rawBodies.rbTranslation(pid);
      const rotation = rawBodies.rbRotation(pid);
      const linvel = rawBodies.rbLinvel(pid);

      RigidBodyState.update(
        eid,
        translation.x,
        translation.y,
        rotation.angle,
        linvel.x,
        linvel.y,
        rawBodies.rbAngvel(pid),
      );

      translation.free();
      rotation.free();
      linvel.free();
    }
  };
}
