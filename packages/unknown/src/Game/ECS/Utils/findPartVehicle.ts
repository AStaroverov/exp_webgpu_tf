import { EntityId, hasComponent } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { getGameComponents } from "../createGameWorld.ts";

/**
 * Resolves a vehicle part to its owning vehicle, or `undefined` when the part
 * is no longer on one (torn-off debris with a stale parent chain). Mirrors the
 * `fillSlot` lookup: the part's slot is parented to either the vehicle (hull
 * parts) or the turret (turret parts) — one extra hop for the latter. The
 * result is VERIFIED to be a `Vehicle`; never a guessed eid.
 */
export function findVehicleEidByPartEid(
  partEid: EntityId,
  { world } = GameDI,
): EntityId | undefined {
  const { Parent, Vehicle } = getGameComponents(world);
  const mid = Parent.id[Parent.id[partEid]] as EntityId; // slot's parent: vehicle (hull) OR turret
  if (hasComponent(world, mid, Vehicle)) return mid;

  const top = Parent.id[mid] as EntityId; // turret part → one hop higher
  return hasComponent(world, top, Vehicle) ? top : undefined;
}
