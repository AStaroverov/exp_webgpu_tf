import { addComponent, observe, onAdd, onRemove } from "bitecs";
import type { World } from "bitecs";
import { createTable } from "./Table.ts";
import type { Table, TableHandle } from "./Table.ts";

const $CompRef = Symbol("CompRef");
let indexCompRef = 0;
let nextCompRef: any = { [$CompRef]: indexCompRef++ };

type ReactiveSetter = (eid: number, ...args: any[]) => any;

export type Obs = <F extends ReactiveSetter>(setter: F) => F;

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
    };
    const comp = Object.assign(ref, create(ref, ctx)) as T;
    nextCompRef = { [$CompRef]: indexCompRef++ };
    return comp;
  };
}
