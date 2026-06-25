/**
 * Sparse-set table storage for sparse components (carriers ≪ maxEntities).
 *
 * One component = one table = one chunked (paged) sparse index `eid → row`.
 * The dense part is the row storage shared by all columns: a column is a
 * typed array indexed by row, registered via `table.flat(Ctor)` (scalar per
 * eid) or `table.nested(Ctor, batchLength)` (strided vector per eid). Removing
 * a row is swap-and-pop over ALL columns at once, so columns can never
 * desynchronize.
 *
 * Public surface (what a component sees) is `Table` — only the column
 * factories; a column exposes only eid-keyed accessors. Row creation/removal
 * is NOT public: it lives on the `TableHandle` returned by `createTable`,
 * consumed exclusively by the entity-lifecycle hooks in `defineComponent`
 * (`onAdd` → ensureRow, `onRemove` → removeRow). The ECS stays the single
 * source of truth for component presence:
 * - presence — `hasComponent`;
 * - iteration over carriers — `query`;
 * - removal — `removeComponent` / `removeEntity` (the table cleans itself).
 *
 * ZAII: `get` of an absent row returns 0; creating a row zeroes every column.
 * `set` of an absent row throws — writing outside an `addComponent`-created
 * row would fork the truth about presence away from the ECS.
 */

const PAGE_SIZE = 4096; // EnTT's default sparse page; shift/mask, 16 KB per Uint32 page
const PAGE_SHIFT = 12;
const PAGE_MASK = PAGE_SIZE - 1;
const TOMBSTONE = 0xffffffff;
const INITIAL_CAPACITY = 64;

type ColumnArray =
  | Float64Array
  | Float32Array
  | Int32Array
  | Uint32Array
  | Int16Array
  | Uint16Array
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray;

export type ColumnCtor<T extends ColumnArray = ColumnArray> = new (length: number) => T;

// Structural view of a typed array used by the dense storage internally:
// method calls on the ColumnArray union don't reduce (TS intersects the
// signatures), a common shape sidesteps that without casts.
type ColumnData = {
  [row: number]: number;
  readonly length: number;
  set(values: ArrayLike<number>, offset?: number): void;
  fill(value: number, start?: number, end?: number): unknown;
  copyWithin(target: number, start: number, end?: number): unknown;
};

export type FlatColumn = {
  get(eid: number): number;
  set(eid: number, value: number): void;
};

/**
 * Strided per-eid vector column — the table-backed counterpart of `NestedArray`
 * (`batchLength` numbers per entity). `getBatch` COPIES into the caller's
 * out-array: a subarray view into the dense storage would silently go stale
 * on swap-remove and on growth.
 */
export type NestedColumn = {
  get(eid: number, index: number): number;
  set(eid: number, index: number, value: number): void;
  setBatch(eid: number, values: ArrayLike<number>): void;
  getBatch<O extends ColumnArray>(eid: number, out: O): O;
};

export type Table = {
  flat(Ctor: ColumnCtor): FlatColumn;
  nested(Ctor: ColumnCtor, batchLength: number): NestedColumn;
};

export type TableHandle = {
  table: Table;
  /** Idempotent; creates a zeroed row. Called ONLY from the onAdd lifecycle hook. */
  ensureRow(eid: number): void;
  /** Swap-and-pop over all columns. Called ONLY from the onRemove lifecycle hook. */
  removeRow(eid: number): void;
};

export function createTable(): TableHandle {
  const pages: (Uint32Array | undefined)[] = [];
  let dense = new Uint32Array(INITIAL_CAPACITY);
  let capacity = INITIAL_CAPACITY;
  let size = 0;
  const columns: { data: ColumnData; Ctor: ColumnCtor; batchLength: number }[] = [];

  // Last-lookup cache. Invalidated on removeRow: swap-and-pop moves a foreign
  // row onto the cached position, so a stale hit would read someone else's data.
  let cachedEid = TOMBSTONE;
  let cachedRow = 0;

  const rowOf = (eid: number): number => {
    if (eid === cachedEid) return cachedRow;
    const page = pages[eid >>> PAGE_SHIFT];
    const row = page === undefined ? TOMBSTONE : page[eid & PAGE_MASK];
    if (row !== TOMBSTONE) {
      cachedEid = eid;
      cachedRow = row;
    }
    return row;
  };

  const pageOf = (eid: number): Uint32Array => {
    const index = eid >>> PAGE_SHIFT;
    let page = pages[index];
    if (page === undefined) {
      page = new Uint32Array(PAGE_SIZE).fill(TOMBSTONE);
      pages[index] = page;
    }
    return page;
  };

  const grow = () => {
    capacity *= 2;
    const nextDense = new Uint32Array(capacity);
    nextDense.set(dense);
    dense = nextDense;
    for (let i = 0; i < columns.length; i++) {
      const column = columns[i];
      const next: ColumnData = new column.Ctor(capacity * column.batchLength);
      next.set(column.data);
      column.data = next;
    }
  };

  const requireRow = (eid: number): number => {
    const row = rowOf(eid);
    if (row === TOMBSTONE) {
      throw new Error(`Table: set(${eid}) on an entity without the component`);
    }
    return row;
  };

  const table: Table = {
    flat(Ctor: ColumnCtor): FlatColumn {
      const column = { data: new Ctor(capacity), Ctor, batchLength: 1 };
      columns.push(column);
      return {
        get(eid: number): number {
          const row = rowOf(eid);
          return row === TOMBSTONE ? 0 : column.data[row];
        },
        set(eid: number, value: number): void {
          column.data[requireRow(eid)] = value;
        },
      };
    },
    nested(Ctor: ColumnCtor, batchLength: number): NestedColumn {
      const column = { data: new Ctor(capacity * batchLength), Ctor, batchLength };
      columns.push(column);
      return {
        get(eid: number, index: number): number {
          const row = rowOf(eid);
          return row === TOMBSTONE ? 0 : column.data[row * batchLength + index];
        },
        set(eid: number, index: number, value: number): void {
          column.data[requireRow(eid) * batchLength + index] = value;
        },
        setBatch(eid: number, values: ArrayLike<number>): void {
          column.data.set(values, requireRow(eid) * batchLength);
        },
        getBatch<O extends ColumnArray>(eid: number, out: O): O {
          const row = rowOf(eid);
          if (row === TOMBSTONE) {
            out.fill(0);
            return out;
          }
          const start = row * batchLength;
          for (let i = 0; i < batchLength; i++) {
            out[i] = column.data[start + i];
          }
          return out;
        },
      };
    },
  };

  return {
    table,
    ensureRow(eid: number): void {
      const page = pageOf(eid);
      if (page[eid & PAGE_MASK] !== TOMBSTONE) return;
      if (size === capacity) grow();
      const row = size++;
      page[eid & PAGE_MASK] = row;
      dense[row] = eid;
      for (let i = 0; i < columns.length; i++) {
        const { data, batchLength } = columns[i];
        data.fill(0, row * batchLength, (row + 1) * batchLength);
      }
    },
    removeRow(eid: number): void {
      const page = pages[eid >>> PAGE_SHIFT];
      if (page === undefined) return;
      const row = page[eid & PAGE_MASK];
      if (row === TOMBSTONE) return;
      const last = --size;
      if (row !== last) {
        const movedEid = dense[last];
        dense[row] = movedEid;
        pages[movedEid >>> PAGE_SHIFT]![movedEid & PAGE_MASK] = row;
        for (let i = 0; i < columns.length; i++) {
          const { data, batchLength } = columns[i];
          data.copyWithin(row * batchLength, last * batchLength, (last + 1) * batchLength);
        }
      }
      page[eid & PAGE_MASK] = TOMBSTONE;
      cachedEid = TOMBSTONE;
    },
  };
}
