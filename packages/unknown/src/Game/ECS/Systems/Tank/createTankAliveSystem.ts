import { GameDI } from "../../../DI/GameDI.ts";
import { query } from "bitecs";
import { destroyTank, getTankHealth } from "../../Entities/Tank/TankUtils.ts";
import { getGameComponents } from "../../createGameWorld.ts";

export function createTankAliveSystem({ world } = GameDI) {
  const { Vehicle, Children } = getGameComponents(world);

  return () => {
    const vehicleEids = query(world, [Vehicle, Children]);

    for (const vehicleEid of vehicleEids) {
      const hp = getTankHealth(vehicleEid);
      hp === 0 && destroyTank(vehicleEid);
    }
  };
}
