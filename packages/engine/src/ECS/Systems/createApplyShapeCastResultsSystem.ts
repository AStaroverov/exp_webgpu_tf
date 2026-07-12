import { entityExists, hasComponent } from "bitecs";
import type { EngineWorld } from "../createEngineWorld.ts";
import { getEngineComponents, getEngineSab } from "../createEngineWorld.ts";

// CONSUMER half of the shape-cast query: drain the HITS ring (worker → main) and
// land each record on its caster's per-entity hit ring (ShapeCaster.appendHit owns
// the queryId grouping). A caster that despawned or dropped its ShapeCaster while
// the query was in flight simply discards the record — absence is handled here,
// never guessed downstream.
export function createApplyShapeCastResultsSystem(world: EngineWorld): () => void {
  const { ShapeCaster } = getEngineComponents(world);
  const sab = getEngineSab(world);

  return function applyShapeCastResults() {
    if (!sab.isProducer) return;
    sab.drainHits((queryId, casterEid, hitEid) => {
      if (!entityExists(world, casterEid)) return;
      if (!hasComponent(world, casterEid, ShapeCaster)) return;
      ShapeCaster.appendHit(casterEid, queryId, hitEid);
    });
  };
}
