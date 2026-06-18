import { addComponent, query } from "bitecs";
import type { World } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { getGameComponents } from "../createGameWorld.ts";
import { defineComponent } from "renderer/src/ECS/utils.ts";

export const createTeamRefComponent = defineComponent((TeamRef, ctx) => {
  const id = ctx.table.flat(Uint32Array);
  return {
    id,
    addComponent(world: World, eid: number, team: number) {
      addComponent(world, eid, TeamRef);
      id.set(eid, team);
    },
  };
});

export function getTeamsCount({ world } = GameDI) {
  const { Tank, TeamRef } = getGameComponents(world);
  const tanks = query(world, [Tank]);
  const teamsCount = new Set(tanks.map((tankId) => TeamRef.id.get(tankId)));
  return teamsCount.size;
}
