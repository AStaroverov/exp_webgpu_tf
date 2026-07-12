import { addComponent, World } from "bitecs";
import { defineComponent } from "../../../../common/src/component.ts";

// A capsule world-query attached to an entity: every frame the entity carries this
// component, the engine sweeps a capsule spanning the localFrom→localTo segment
// (expressed in the ENTITY'S LOCAL space, e.g. hilt→tip of a blade) transformed by
// its GlobalTransform, and the physics worker answers with every body it overlaps.
//
// This is what decouples hit detection from animation: animation only writes
// LocalTransform matrices as it already does; the caster only reads the resulting
// world matrix. Whoever decides an attack is happening adds the component for the
// strike's active window and removes it after (existence-based processing).
//
// Results land back on the entity (per-entity ring, one physics step later):
// `resultQueryId` stamps which cast the ring holds, so a new cast's first record
// resets the ring instead of mixing with the previous frame's hits. excludeEid
// names the body that must not self-hit (the wielder's collider) — the caster
// itself usually has no body. 0 = exclude nothing.

export const SHAPE_CASTER_MAX_HITS = 16;

export type ShapeCasterSpec = {
  readonly localFrom: readonly [number, number, number];
  readonly localTo: readonly [number, number, number];
  readonly radius: number;
  readonly excludeEid?: number;
};

export const createShapeCasterComponent = defineComponent((ShapeCaster, ctx) => {
  const localFrom = ctx.table.nested(Float64Array, 3);
  const localTo = ctx.table.nested(Float64Array, 3);
  const radius = ctx.table.flat(Float64Array);
  const excludeEid = ctx.table.flat(Float64Array);
  // Written by the engine systems, read by gameplay through the accessors below.
  const queryId = ctx.table.flat(Float64Array); // last issued cast
  const resultQueryId = ctx.table.flat(Float64Array); // cast the hit ring holds
  const hitCount = ctx.table.flat(Float64Array);
  const hits = ctx.table.nested(Float64Array, SHAPE_CASTER_MAX_HITS);

  return {
    localFrom,
    localTo,
    radius,
    excludeEid,
    queryId,
    addComponent(world: World, eid: number, spec: ShapeCasterSpec) {
      addComponent(world, eid, ShapeCaster);
      localFrom.setBatch(eid, spec.localFrom);
      localTo.setBatch(eid, spec.localTo);
      radius.set(eid, spec.radius);
      excludeEid.set(eid, spec.excludeEid ?? 0);
      queryId.set(eid, 0);
      resultQueryId.set(eid, 0);
      hitCount.set(eid, 0);
    },
    getHitCount(eid: number): number {
      return hitCount.get(eid);
    },
    getHit(eid: number, index: number): number {
      return hits.get(eid, index);
    },
    // Route one HITS-ring record onto its caster (result system only). A record
    // stamped with a NEWER queryId starts that cast's batch — the ring is reset.
    appendHit(eid: number, recordQueryId: number, hitEid: number) {
      if (resultQueryId.get(eid) !== recordQueryId) {
        resultQueryId.set(eid, recordQueryId);
        hitCount.set(eid, 0);
      }
      const count = hitCount.get(eid);
      if (count >= SHAPE_CASTER_MAX_HITS) return;
      hits.set(eid, count, hitEid);
      hitCount.set(eid, count + 1);
    },
  };
});
