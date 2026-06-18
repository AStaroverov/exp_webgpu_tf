import { addEntity, EntityId } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { getGameComponents } from "../createGameWorld.ts";

export function createPlayer(teamId: number, { world } = GameDI): EntityId {
  const { TeamRef } = getGameComponents(world);
  const playerId = addEntity(world);
  TeamRef.addComponent(world, playerId, teamId);
  return playerId;
}
