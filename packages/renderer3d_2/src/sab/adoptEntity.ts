// Force a bitecs world to ADOPT a caller-chosen eid (plan §4.2; proven in
// spikes/eid-adoption.mjs). The shared CONTROL-SAB NEXT_EID counter is the one
// and only eid authority for both the render (main) and physics (worker) worlds;
// each world must then materialize that exact eid. bitecs 0.4 has no public
// addEntityWithId, so we use a small custom id-path over the documented, exported
// `$internal` shape with versioning:false (eid === raw id).
//
// Coupling (stable across bitecs 0.4, present in the published .d.ts):
//   - $internal Symbol → WorldContext { entityIndex, entityComponents, notQueries }
//   - EntityIndex { dense, sparse, aliveCount, maxId, versioning }
// This replays everything addEntity() does EXCEPT the recycle branch (we never
// recycle) and the internal notQueries refresh (inert: no Not() queries exist on
// bridge entities — see the documented gap in the spike).

import { $internal } from "bitecs";
import type { World } from "bitecs";

type EntityIndex = {
  aliveCount: number;
  dense: number[];
  sparse: number[];
  maxId: number;
  versioning: boolean;
};
type WorldContext = {
  entityIndex: EntityIndex;
  entityComponents: Map<number, Set<unknown>>;
};

function isAlive(index: EntityIndex, eid: number): boolean {
  const di = index.sparse[eid];
  return di !== undefined && di < index.aliveCount && index.dense[di] === eid;
}

export function adoptEntity(world: World, eid: number): number {
  const ctx = (world as unknown as { [$internal]: WorldContext })[$internal];
  const index = ctx.entityIndex;

  if (index.versioning) {
    throw new Error("adoptEntity requires versioning:false (raw eid === id)");
  }
  if (isAlive(index, eid)) {
    throw new Error(`adoptEntity: eid ${eid} already alive in this world (double adopt)`);
  }

  // Insert eid as a live dense/sparse pair — same structure addEntityId builds,
  // minus the recycle branch.
  const denseIndex = index.aliveCount;
  index.dense[denseIndex] = eid;
  index.sparse[eid] = denseIndex;
  index.aliveCount++;
  // Keep maxId monotonic so any stray internal addEntity can never collide.
  if (eid > index.maxId) index.maxId = eid;

  // Replay addEntity()'s per-entity init so queries/addComponent work.
  ctx.entityComponents.set(eid, new Set());

  return eid;
}
