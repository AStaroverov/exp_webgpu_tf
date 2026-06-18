// Запуск: node --experimental-transform-types --test src/Game/ECS/Components/Dot.test.ts
// (transform-types — из-за enum DamageKind; системные модули не импортируются,
// они тянут rapier-wasm через GameDI)
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { addEntity, createWorld, hasComponent, query, removeComponent, removeEntity } from "bitecs";
import { createDotComponent } from "./Dot.ts";
import { createStunnedComponent } from "./Stunned.ts";
import { DamageKind } from "./Damagable.ts";

// Дословно проход createExpirySystem: обратная итерация query, removeComponent на нуле.
const tickExpiry = (world: object, component: { tick(eid: number, d: number): boolean }, delta: number) => {
  const eids = query(world, [component]);
  for (let i = eids.length - 1; i >= 0; i--) {
    if (component.tick(eids[i], delta)) {
      removeComponent(world, eids[i], component);
    }
  }
};

describe("Dot pilot on the sparse-set table", () => {
  it("refresh stamps, extends to max and keeps the strongest dps", () => {
    const world = createWorld({});
    const Dot = createDotComponent(world);
    const eid = addEntity(world);

    Dot.refresh(eid, DamageKind.Fire, 30, 2000);
    assert.equal(hasComponent(world, eid, Dot), true);
    assert.equal(Dot.dps.get(eid), 30);
    assert.equal(Dot.kind.get(eid), DamageKind.Fire);
    assert.equal(Dot.remainingMs.get(eid), 2000);

    Dot.refresh(eid, DamageKind.Fire, 30, 500); // max, not stack
    assert.equal(Dot.remainingMs.get(eid), 2000);
    assert.equal(Dot.getRemainingFraction(eid), 1);
  });

  it("expiry pass removes at 0; the table cleans itself; re-stamp reads fresh", () => {
    const world = createWorld({});
    const Dot = createDotComponent(world);
    const a = addEntity(world);
    const b = addEntity(world);

    Dot.refresh(a, DamageKind.Fire, 10, 1000);
    Dot.refresh(b, DamageKind.Fire, 20, 3000);

    tickExpiry(world, Dot, 1000); // a истёк, b жив — удаление внутри итерации query
    assert.equal(hasComponent(world, a, Dot), false);
    assert.equal(hasComponent(world, b, Dot), true);
    assert.equal(Dot.dps.get(a), 0); // строка снята хуком, ZAII
    assert.equal(Dot.dps.get(b), 20);

    Dot.refresh(a, DamageKind.Physical, 5, 100); // повторный стамп после истечения
    assert.equal(Dot.dps.get(a), 5);
    assert.equal(Dot.kind.get(a), DamageKind.Physical);
    assert.equal(Dot.remainingMs.get(a), 100);
  });

  it("removeEntity drops the dot; a recycled eid reads clean", () => {
    const world = createWorld({});
    const Dot = createDotComponent(world);
    const eid = addEntity(world);
    Dot.refresh(eid, DamageKind.Fire, 50, 9000);

    removeEntity(world, eid);
    const reused = addEntity(world);
    assert.equal(reused, eid);
    assert.equal(Dot.dps.get(reused), 0);
    assert.equal(Dot.remainingMs.get(reused), 0);
  });

  it("Stunned shares the expiry sub-component but owns its table", () => {
    const world = createWorld({});
    const Dot = createDotComponent(world);
    const Stunned = createStunnedComponent(world);
    const eid = addEntity(world);

    Dot.refresh(eid, DamageKind.Fire, 10, 1000);
    Stunned.refresh(eid, 400);
    assert.equal(Dot.remainingMs.get(eid), 1000);
    assert.equal(Stunned.remainingMs.get(eid), 400);

    tickExpiry(world, Stunned, 500);
    assert.equal(hasComponent(world, eid, Stunned), false);
    assert.equal(hasComponent(world, eid, Dot), true);
    assert.equal(Dot.remainingMs.get(eid), 1000); // не задет чужим удалением
  });
});
