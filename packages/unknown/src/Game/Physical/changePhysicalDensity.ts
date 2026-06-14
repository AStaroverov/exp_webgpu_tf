import { GameDI } from "../DI/GameDI.ts";
import { getGameComponents } from "../ECS/createGameWorld.ts";

export function changePhysicalDensity(
  eid: number,
  density: number,
  { world, physicalWorld } = GameDI,
) {
  const { RigidBodyRef } = getGameComponents(world);
  const physicalId = RigidBodyRef.id[eid];
  const rigidBody = physicalWorld.getRigidBody(physicalId);

  if (rigidBody == null) return;

  for (let i = 0; i < rigidBody.numColliders(); i++) {
    rigidBody.collider(i).setDensity(density);
  }
}
