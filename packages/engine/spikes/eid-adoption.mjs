// THROWAWAY SPIKE 0d — shared-counter eid adoption across two bitecs worlds.
//
// Proves §4.2 of docs/physics-worker-sab-plan.md: ONE monotonic counter is the
// sole eid authority; both worlds adopt the SAME eid; nothing is recycled; the
// two worlds never diverge even when an entity exists in only one of them.
//
// NODE-only, single process, no WASM, no worker. Run:
//   node packages/engine/spikes/eid-adoption.mjs
//
// ---------------------------------------------------------------------------
// bitecs 0.4 reality (verified in node_modules/bitecs/dist/core/index.mjs):
//   - The PUBLIC export `addEntity(world)` always calls the INTERNAL
//     `addEntityId(index)` which (a) recycles from `dense[aliveCount]` if any
//     id is free, else (b) does `++index.maxId`. It takes NO eid argument and
//     gives you NO way to force a specific id. (The plan's assumed
//     `addEntityId(ctx.entityIndex, eid)` two-arg signature DOES NOT EXIST.)
//   - `addEntityId` / `removeEntityId` / `isEntityIdAlive` are NOT re-exported
//     from the package entry (index.d.ts only re-exports createEntityIndex,
//     getId, getVersion, withVersioning). So there is no near-public forcing
//     call either. The escape hatch is the `$internal` symbol, which IS
//     exported and gives us the WorldContext incl. `entityIndex`.
//
// => Chosen mechanism: a small CUSTOM id-path that writes the externally-chosen
//    eid straight into `ctx.entityIndex` (dense/sparse/maxId/aliveCount) with
//    `versioning:false`, then replays the rest of what addEntity() does so the
//    entity is a first-class citizen (entityExists === true, addComponent and
//    queries work). versioning:false keeps eid === raw id (no version bits
//    packed in), which is what the shared counter hands out.
// ---------------------------------------------------------------------------

import {
  $internal,
  createWorld,
  createEntityIndex,
  withVersioning,
  removeEntity,
  entityExists,
  addComponent,
  hasComponent,
  query,
} from "bitecs";

let failures = 0;
const assert = (cond, msg) => {
  if (cond) {
    console.log("  ok  -", msg);
  } else {
    failures++;
    console.log("  FAIL-", msg);
  }
};

// --- 2. The shared monotonic counter --------------------------------------
// In this single-process spike it is a plain `let`. In production it is an
// Int32 slot in the CONTROL SharedArrayBuffer and `nextEid()` becomes
//   Atomics.add(CONTROL, NEXT_EID, 1)
// (atomic, lock-free, the one and only id authority for BOTH threads).
//
// NOTE: bitecs uses 0 as a valid first id only via ++maxId (ids start at 1).
// We mirror that — counter starts at 1 — so adopted ids match what a lone
// bitecs world would have produced, keeping the layout intuition identical.
let NEXT_EID = 1;
const handedOut = new Set(); // audit: every id the counter has ever emitted
function nextEid() {
  const eid = NEXT_EID++; // prod: Atomics.add(CONTROL, NEXT_EID, 1)
  if (handedOut.has(eid)) {
    throw new Error(`counter re-handed-out eid ${eid} — monotonicity broken`);
  }
  handedOut.add(eid);
  return eid;
}

// --- 3. Force a world to ADOPT a specific eid ------------------------------
// Custom id-path: replicate addEntity()'s bookkeeping but with a caller-chosen
// id instead of letting addEntityId() allocate one. Couples to the documented
// EntityIndex shape {dense, sparse, aliveCount, maxId} + the WorldContext
// {entityIndex, entityComponents, notQueries}. All reached via $internal.
function adoptEntity(world, eid) {
  const ctx = world[$internal];
  const index = ctx.entityIndex;

  if (index.versioning) {
    throw new Error("adoptEntity requires versioning:false (raw eid === id)");
  }
  if (isAlive(index, eid)) {
    throw new Error(`eid ${eid} already alive in this world — double adopt`);
  }

  // Insert `eid` as a live dense/sparse pair (same structure addEntityId builds,
  // minus the recycle branch — we never recycle).
  const denseIndex = index.aliveCount;
  index.dense[denseIndex] = eid;
  index.sparse[eid] = denseIndex;
  index.aliveCount++;
  // Keep maxId monotonic so any later *internal* addEntity (we don't use it,
  // but be safe) can never collide with an adopted id.
  if (eid > index.maxId) index.maxId = eid;

  // Replay the rest of addEntity()'s per-entity init so queries/components work.
  ctx.entityComponents.set(eid, new Set());
  // notQueries: an entity with no components may match a Not(...) query.
  ctx.notQueries.forEach((q) => {
    // queryCheckEntity/queryAddEntity are internal & unexported; in this spike
    // we use no Not() queries, so this set is empty and the omission is inert.
    // Documented coupling: if prod ever uses Not() queries, route adoption
    // through addEntity-equivalent internals or accept that Not() membership is
    // refreshed lazily on next query commit.
    void q;
  });

  return eid;
}

// Local copy of bitecs's isEntityIdAlive logic (unexported) — versioning:false
// so getId(id) === id.
function isAlive(index, eid) {
  const di = index.sparse[eid];
  return di !== undefined && di < index.aliveCount && index.dense[di] === eid;
}

// Convenience: pull a fresh shared id and adopt it on ONE world.
function spawnIn(world, label, eid = nextEid()) {
  adoptEntity(world, eid);
  return eid;
}

// --- 1. Two worlds in one process -----------------------------------------
// versioning:false is mandatory for the shared-counter model: with versioning
// on, bitecs packs a version into the high bits and the "eid" a query yields is
// NOT the raw counter value. We want eid === counter value, identically on both.
// createEntityIndex() with NO args defaults to versioning:false (verified in
// bitecs source line 22). withVersioning is imported only to document that we
// deliberately do NOT use it.
void withVersioning;
const renderWorld = createWorld(createEntityIndex());
const physicsWorld = createWorld(createEntityIndex());

// A bridge component (lives in SAB in prod; here just a marker so we can prove
// presence/queries survive adoption). In prod this is RigidBodyRef etc.
const RigidBodyRef = {}; // bitecs treats any object as a component ref.

console.log("\n== Scenario ==");
console.log("counter is the sole authority; render creates render-only ents,");
console.log("physics creates body ents; both adopt from the same counter.\n");

// Render world creates: a body entity (also in physics) + two render-only
// entities (lights) that physics NEVER sees.
console.log("[render] adopt body eid + 2 render-only lights");
const bodyEid = spawnIn(renderWorld, "render"); // 1 — a physical body, render side
const lightA = spawnIn(renderWorld, "render");  // 2 — render-only
const lightB = spawnIn(renderWorld, "render");  // 3 — render-only

// Physics world adopts the SAME body eid (told over the op channel in prod).
console.log("[physics] adopt the SAME body eid (1)");
adoptEntity(physicsWorld, bodyEid);
addComponent(physicsWorld, bodyEid, RigidBodyRef);
addComponent(renderWorld, bodyEid, RigidBodyRef);

// Physics world creates a NEW body that render will also adopt.
console.log("[physics] adopt a fresh body eid (4); render adopts same");
const body2 = spawnIn(physicsWorld, "physics"); // 4
adoptEntity(renderWorld, body2);

console.log("\n== Assertions: identical meaning + no divergence ==");
assert(bodyEid === 1, `body eid is first counter value (got ${bodyEid})`);
assert(lightA === 2 && lightB === 3, "render-only lights consumed counter values 2,3");
assert(body2 === 4, `physics body pulled fresh value 4 (got ${body2})`);

assert(entityExists(renderWorld, bodyEid) && entityExists(physicsWorld, bodyEid),
  "eid 1 (body) exists in BOTH worlds");
assert(entityExists(renderWorld, lightA) && !entityExists(physicsWorld, lightA),
  "eid 2 (light) exists ONLY in render world");
assert(entityExists(renderWorld, lightB) && !entityExists(physicsWorld, lightB),
  "eid 3 (light) exists ONLY in render world");
assert(entityExists(renderWorld, body2) && entityExists(physicsWorld, body2),
  "eid 4 (body2) exists in BOTH worlds");

assert(hasComponent(renderWorld, bodyEid, RigidBodyRef) &&
       hasComponent(physicsWorld, bodyEid, RigidBodyRef),
  "RigidBodyRef present on eid 1 in both worlds (component ops work post-adopt)");
assert(!hasComponent(physicsWorld, lightA, RigidBodyRef),
  "light A has no body component (and physics doesn't even know it)");

// queries operate over adopted entities
const physBodies = query(physicsWorld, [RigidBodyRef]);
assert(physBodies.includes(bodyEid),
  "query([RigidBodyRef]) in physics world returns the adopted body");

console.log("\n== Removal: prove NO recycle by the authority ==");
// Remove the body from BOTH worlds (fire-and-forget order doesn't matter).
const removedEid = bodyEid;
removeEntity(renderWorld, removedEid);
removeEntity(physicsWorld, removedEid);
assert(!entityExists(renderWorld, removedEid) && !entityExists(physicsWorld, removedEid),
  `eid ${removedEid} removed from both worlds`);

// Also remove a render-only light to exercise the lone-world removal path.
removeEntity(renderWorld, lightA);
assert(!entityExists(renderWorld, lightA), "render-only light A removed");

// CRITICAL: a subsequent create pulls a FRESH counter value, NOT a recycled id.
// (bitecs's own addEntity WOULD recycle here via dense[aliveCount]; the counter
//  authority bypasses that path entirely.)
const afterRemoval = spawnIn(renderWorld, "render"); // must be 5, never 1/2
adoptEntity(physicsWorld, afterRemoval);
assert(afterRemoval === 5,
  `post-removal create got FRESH value 5, not a recycled id (got ${afterRemoval})`);
assert(afterRemoval !== removedEid && afterRemoval !== lightA,
  "post-removal eid is none of the just-removed eids");
const priorMax = Math.max(...[...handedOut].filter((e) => e !== afterRemoval));
assert(afterRemoval > priorMax,
  "new eid strictly greater than every previously handed-out eid (monotonic)");

console.log("\n== Stress: interleaved lone/shared spawns + removes, no reuse ==");
// Randomized-ish sequence to make sure the two worlds never diverge and the
// counter never re-emits a value, regardless of which world originates.
const aliveBoth = new Set([body2, afterRemoval]); // eids present in both worlds
const aliveRenderOnly = new Set([lightB]);
let diverged = false;

for (let step = 0; step < 2000; step++) {
  const roll = step % 5;
  if (roll === 0) {
    // shared spawn (a body): originate anywhere, adopt on both
    const e = nextEid();
    adoptEntity(renderWorld, e);
    adoptEntity(physicsWorld, e);
    aliveBoth.add(e);
  } else if (roll === 1) {
    // render-only spawn (a light): physics never sees it
    const e = nextEid();
    adoptEntity(renderWorld, e);
    aliveRenderOnly.add(e);
  } else if (roll === 2 && aliveBoth.size > 1) {
    // shared despawn
    const e = aliveBoth.values().next().value;
    aliveBoth.delete(e);
    removeEntity(renderWorld, e);
    removeEntity(physicsWorld, e);
  } else if (roll === 3 && aliveRenderOnly.size > 0) {
    // render-only despawn
    const e = aliveRenderOnly.values().next().value;
    aliveRenderOnly.delete(e);
    removeEntity(renderWorld, e);
  } else {
    // physics-originated shared spawn
    const e = nextEid();
    adoptEntity(physicsWorld, e);
    adoptEntity(renderWorld, e);
    aliveBoth.add(e);
  }

  // invariant: every "both" eid exists in both; every render-only exists only
  // in render; physics never knows render-only eids.
  for (const e of aliveBoth) {
    if (!entityExists(renderWorld, e) || !entityExists(physicsWorld, e)) {
      diverged = true;
    }
  }
  for (const e of aliveRenderOnly) {
    if (!entityExists(renderWorld, e) || entityExists(physicsWorld, e)) {
      diverged = true;
    }
  }
  if (diverged) break;
}
assert(!diverged, "after 2000 interleaved ops: no world divergence");

// Final monotonicity check: handedOut is exactly 1..NEXT_EID-1 with no gaps and
// no duplicates (Set size == count proves no value was emitted twice).
assert(handedOut.size === NEXT_EID - 1,
  `counter emitted ${handedOut.size} unique ids == NEXT_EID-1 (${NEXT_EID - 1}); none recycled`);
let contiguous = true;
for (let i = 1; i < NEXT_EID; i++) if (!handedOut.has(i)) contiguous = false;
assert(contiguous, "every value 1..NEXT_EID-1 was handed out exactly once (strict monotonic)");

// Prove dead eids are never re-handed-out: none of the removed eids can appear
// again because nextEid() only ever moves forward.
const everRemoved = [removedEid, lightA]; // explicit removals above
assert(everRemoved.every((e) => {
  // it WAS handed out once, and the counter is now strictly past it
  return handedOut.has(e) && e < NEXT_EID;
}), "removed eids stay in history and are strictly below the live cursor (never re-handed-out)");

console.log("\n========================================");
console.log(failures === 0 ? "SPIKE 0d: ALL GREEN" : `SPIKE 0d: ${failures} FAILURE(S)`);
console.log(`counter handed out ${handedOut.size} eids total; final NEXT_EID=${NEXT_EID}`);
console.log("========================================");
process.exit(failures === 0 ? 0 : 1);
