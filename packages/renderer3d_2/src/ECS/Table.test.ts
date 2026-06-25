import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { addComponent, addEntity, createWorld, hasComponent, removeComponent, removeEntity } from "bitecs";
import { createTable } from "./Table.ts";
import { defineComponent } from "./utils.ts";

describe("Table core", () => {
  it("get/set roundtrip and ZAII for absent rows", () => {
    const { table, ensureRow } = createTable();
    const dps = table.flat(Float64Array);
    const kind = table.flat(Int8Array);

    assert.equal(dps.get(123), 0);

    ensureRow(7);
    dps.set(7, 1.5);
    kind.set(7, 3);
    assert.equal(dps.get(7), 1.5);
    assert.equal(kind.get(7), 3);
    assert.equal(dps.get(8), 0);
  });

  it("set on an absent row throws", () => {
    const { table } = createTable();
    const col = table.flat(Float64Array);
    assert.throws(() => col.set(5, 1), /without the component/);
  });

  it("row creation zeroes every column (stale dense slot after remove)", () => {
    const { table, ensureRow, removeRow } = createTable();
    const a = table.flat(Float64Array);
    const b = table.flat(Int32Array);

    ensureRow(1);
    a.set(1, 11);
    b.set(1, 12);
    removeRow(1);
    ensureRow(2); // reuses dense row 0
    assert.equal(a.get(2), 0);
    assert.equal(b.get(2), 0);
    assert.equal(a.get(1), 0);
  });

  it("swap-and-pop keeps all columns consistent", () => {
    const { table, ensureRow, removeRow } = createTable();
    const a = table.flat(Float64Array);
    const b = table.flat(Uint16Array);

    for (const eid of [10, 20, 30]) {
      ensureRow(eid);
      a.set(eid, eid + 0.5);
      b.set(eid, eid * 2);
    }
    removeRow(10); // row of 30 swaps into row 0
    assert.equal(a.get(10), 0);
    assert.equal(b.get(10), 0);
    assert.equal(a.get(20), 20.5);
    assert.equal(b.get(20), 40);
    assert.equal(a.get(30), 30.5);
    assert.equal(b.get(30), 60);
  });

  it("lookup cache is invalidated by swap-remove", () => {
    const { table, ensureRow, removeRow } = createTable();
    const col = table.flat(Float64Array);

    ensureRow(100);
    ensureRow(200);
    col.set(100, 1);
    col.set(200, 2);
    assert.equal(col.get(100), 1); // caches eid 100 → row 0
    removeRow(100); // row of 200 swaps onto row 0
    assert.equal(col.get(100), 0);
    assert.equal(col.get(200), 2);
  });

  it("grows past initial capacity and across sparse pages", () => {
    const { table, ensureRow } = createTable();
    const col = table.flat(Float64Array);

    const count = 1000;
    const stride = 4099; // спред по многим страницам (page = 4096)
    for (let i = 0; i < count; i++) {
      const eid = i * stride;
      ensureRow(eid);
      col.set(eid, i + 1);
    }
    for (let i = 0; i < count; i++) {
      assert.equal(col.get(i * stride), i + 1);
    }
  });

  it("nested column: strided get/set/setBatch/getBatch with ZAII", () => {
    const { table, ensureRow } = createTable();
    const pos = table.nested(Float64Array, 2);
    const out = new Float64Array(2);

    assert.deepEqual(Array.from(pos.getBatch(5, out)), [0, 0]);
    assert.equal(pos.get(5, 1), 0);
    assert.throws(() => pos.set(5, 0, 1), /without the component/);
    assert.throws(() => pos.setBatch(5, [1, 2]), /without the component/);

    ensureRow(5);
    pos.setBatch(5, [3, 4]);
    assert.equal(pos.get(5, 0), 3);
    assert.equal(pos.get(5, 1), 4);
    pos.set(5, 1, 7);
    assert.deepEqual(Array.from(pos.getBatch(5, out)), [3, 7]);

    out[0] = 999; // out — копия, мутация не трогает колонку
    assert.equal(pos.get(5, 0), 3);
  });

  it("nested column: swap-remove moves whole batches, row creation zeroes them", () => {
    const { table, ensureRow, removeRow } = createTable();
    const scalar = table.flat(Int32Array);
    const pos = table.nested(Float64Array, 3);
    const out = new Float64Array(3);

    for (const eid of [1, 2, 3]) {
      ensureRow(eid);
      scalar.set(eid, eid * 10);
      pos.setBatch(eid, [eid, eid + 0.25, eid + 0.5]);
    }
    removeRow(1); // batch of 3 swaps into row 0
    assert.deepEqual(Array.from(pos.getBatch(3, out)), [3, 3.25, 3.5]);
    assert.equal(scalar.get(3), 30);
    assert.deepEqual(Array.from(pos.getBatch(2, out)), [2, 2.25, 2.5]);
    assert.deepEqual(Array.from(pos.getBatch(1, out)), [0, 0, 0]);

    ensureRow(1); // reuses the freed dense row
    assert.deepEqual(Array.from(pos.getBatch(1, out)), [0, 0, 0]);
    assert.equal(scalar.get(1), 0);
  });

  it("nested column survives growth", () => {
    const { table, ensureRow } = createTable();
    const pos = table.nested(Float32Array, 2);
    const count = 500;
    for (let eid = 0; eid < count; eid++) {
      ensureRow(eid);
      pos.setBatch(eid, [eid, -eid]);
    }
    for (let eid = 0; eid < count; eid++) {
      assert.equal(pos.get(eid, 0), eid);
      assert.equal(pos.get(eid, 1), -eid);
    }
  });

  it("stress: random add/remove/set matches a Map reference", () => {
    const { table, ensureRow, removeRow } = createTable();
    const cols = [table.flat(Float64Array), table.flat(Int32Array)];
    const refs = [new Map<number, number>(), new Map<number, number>()];

    let seed = 0x2545f491;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed;
    };

    for (let step = 0; step < 50_000; step++) {
      const eid = rand() % 10_000;
      const op = rand() % 3;
      if (op === 0) {
        ensureRow(eid);
        for (let c = 0; c < cols.length; c++) {
          if (!refs[c].has(eid)) refs[c].set(eid, 0);
        }
      } else if (op === 1 && refs[0].has(eid)) {
        const value = rand() % 1000;
        const c = rand() % cols.length;
        cols[c].set(eid, value);
        refs[c].set(eid, value);
      } else if (op === 2) {
        removeRow(eid);
        for (const ref of refs) ref.delete(eid);
      }
      if (step % 1000 === 0) {
        const probe = rand() % 10_000;
        for (let c = 0; c < cols.length; c++) {
          assert.equal(cols[c].get(probe), refs[c].get(probe) ?? 0);
        }
      }
    }
    for (let eid = 0; eid < 10_000; eid++) {
      for (let c = 0; c < cols.length; c++) {
        assert.equal(cols[c].get(eid), refs[c].get(eid) ?? 0);
      }
    }
  });
});

describe("ctx.table lifecycle integration", () => {
  const createTestComponent = defineComponent((_, ctx) => {
    const value = ctx.table.flat(Float64Array);
    return {
      value,
      stamp: (eid: number, v: number) => {
        value.set(eid, v);
      },
    };
  });

  it("addComponent creates a row, removeComponent drops it", () => {
    const world = createWorld({});
    const Comp = createTestComponent(world);
    const eid = addEntity(world);

    addComponent(world, eid, Comp);
    Comp.stamp(eid, 42);
    assert.equal(Comp.value.get(eid), 42);

    removeComponent(world, eid, Comp);
    assert.equal(Comp.value.get(eid), 0);
    assert.throws(() => Comp.stamp(eid, 1));
  });

  it("removeEntity drops the row; a recycled eid reads clean", () => {
    const world = createWorld({});
    const Comp = createTestComponent(world);
    const eid = addEntity(world);
    addComponent(world, eid, Comp);
    Comp.stamp(eid, 99);

    removeEntity(world, eid);
    const reused = addEntity(world);
    assert.equal(reused, eid, "bitecs is expected to recycle the eid immediately");
    assert.equal(Comp.value.get(reused), 0);

    addComponent(world, reused, Comp);
    assert.equal(Comp.value.get(reused), 0, "ZAII: fresh row of a recycled eid is zeroed");
  });

  it("tables of two components are independent", () => {
    const createOther = defineComponent((_, ctx) => {
      const value = ctx.table.flat(Float64Array);
      return { value };
    });
    const world = createWorld({});
    const A = createTestComponent(world);
    const B = createOther(world);
    const eid = addEntity(world);

    addComponent(world, eid, A);
    A.stamp(eid, 5);
    assert.equal(B.value.get(eid), 0);
    assert.equal(hasComponent(world, eid, B as object), false);
  });
});
