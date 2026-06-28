import { addComponent, observe, onAdd, onRemove } from "bitecs";
import type { World } from "bitecs";
import { createTable } from "./Table.ts";
import type { Table, TableHandle } from "./Table.ts";
import type { NestedArray } from "../utils.ts";

const $CompRef = Symbol("CompRef");
let indexCompRef = 0;
let nextCompRef: any = { [$CompRef]: indexCompRef++ };

type ReactiveSetter = (eid: number, ...args: any[]) => any;

export type Obs = <F extends ReactiveSetter>(setter: F) => F;

/**
 * A world's shared-SAB bridge surface, reached via `ctx.sab` — the integration point
 * for SAB-backed (double-buffered) columns, analogous to `ctx.table`. Only worlds that
 * carry a shared SAB (engine physics↔render worlds) provide it; render-only worlds do
 * not, so `ctx.sab` throws there. Byte offsets are resolved by stable NAME from the SAB
 * registry layout (the single source of truth both threads compute identically), so a
 * component never does offset math or receives the SAB as a constructor argument.
 */
export interface ComponentSab {
  /** Last fully-published pose bank (read side, plan §5.2). */
  readBank(): number;
  /** The bank the writer fills before publishing (write side). */
  writeBank(): number;
  /**
   * Per-bank `Float64` views for a registry bridge column, by stable name. Length =
   * the column's bank count (2 for a double-buffered pose, 1 for single-buffered).
   * Built and cached from the registry layout; the component just picks the active bank.
   */
  banks(name: string): NestedArray<Float64ArrayConstructor>[];
}

export type ComponentContext = {
  readonly obs: Obs;
  readonly world: World;
  /**
   * Lazy sparse-set table of THIS component (created and wired to the entity
   * lifecycle on first access): `addComponent` creates a zeroed row,
   * `removeComponent`/`removeEntity` drop it. Sub-components share the parent's
   * ctx, so their columns land in the parent's table automatically.
   */
  readonly table: Table;
  /**
   * The world's shared-SAB bridge (engine worlds only; throws on render-only worlds).
   * Bind SAB-backed double-buffered columns by registry name instead of threading the
   * SAB through the component factory by hand.
   */
  readonly sab: ComponentSab;
};

export type SubComponent<T extends object, A extends unknown[] = []> = (
  component: object,
  ctx: ComponentContext,
  ...args: A
) => T;

const subComponentRegistry = new WeakMap<World, WeakMap<SubComponent<any, any>, object[]>>();

export function getComponents<T extends object>(
  world: World,
  constr: SubComponent<T, any>,
): (T & object)[] {
  return (subComponentRegistry.get(world)?.get(constr) ?? []) as (T & object)[];
}

export function defineSubComponent<T extends object, A extends unknown[] = []>(
  create: SubComponent<T, A>,
): SubComponent<T, A> {
  const constr: SubComponent<T, A> = (component, ctx, ...args) => {
    const byConstr =
      subComponentRegistry.get(ctx.world) ?? new WeakMap<SubComponent<any, any>, object[]>();
    byConstr.set(constr, (byConstr.get(constr) ?? []).concat(component));
    subComponentRegistry.set(ctx.world, byConstr);
    return create(component, ctx, ...args);
  };

  return constr;
}

export function defineComponent<T extends object>(
  create: (ref: object, ctx: ComponentContext) => T,
) {
  return (world: World): T => {
    const ref = nextCompRef as object;
    const localObs: Obs = <F extends ReactiveSetter>(setter: F): F => {
      const setData = { component: ref, data: null };
      return ((eid: number, ...args: Parameters<F> extends [number, ...infer R] ? R : never) => {
        // addComponent BEFORE the write: table-backed columns require the row
        // to exist (set on an absent row throws); change detectors only collect
        // eids, so the notify order within the call doesn't matter.
        addComponent(world, eid, setData);
        return setter(eid, ...args);
      }) as F;
    };
    let tableHandle: TableHandle | null = null;
    const ctx: ComponentContext = {
      obs: localObs,
      world,
      get table(): Table {
        if (tableHandle === null) {
          tableHandle = createTable();
          observe(world, onAdd(ref), tableHandle.ensureRow);
          observe(world, onRemove(ref), tableHandle.removeRow);
        }
        return tableHandle.table;
      },
      // The shared SAB lives on the world context (engine worlds set it before building
      // components). Render-only worlds have none → using ctx.sab there is a bug.
      get sab(): ComponentSab {
        const s = (world as World & { sab?: ComponentSab }).sab;
        if (!s) {
          throw new Error(
            "ctx.sab: this world has no shared SAB (only engine physics/render worlds " +
              "carry one). A render-only component must not use ctx.sab.",
          );
        }
        return s;
      },
    };
    const comp = Object.assign(ref, create(ref, ctx)) as T;
    nextCompRef = { [$CompRef]: indexCompRef++ };
    return comp;
  };
}
