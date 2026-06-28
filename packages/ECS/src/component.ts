import { addComponent, observe, onAdd, onRemove } from "bitecs";
import type { World } from "bitecs";
import { createTable } from "./Table.ts";
import type { Table, TableHandle } from "./Table.ts";
import type { NestedArray } from "./typedArray.ts";

const $CompRef = Symbol("CompRef");
let indexCompRef = 0;
let nextCompRef: any = { [$CompRef]: indexCompRef++ };

type ReactiveSetter = (eid: number, ...args: any[]) => any;

export type Obs = <F extends ReactiveSetter>(setter: F) => F;

export type ComponentContext = {
  readonly obs: Obs;
  readonly world: World;
  readonly table: Table;
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
      get sab(): ComponentSab {
        const s = getComponentSab(world);
        if (s === undefined) {
          throw new Error("ctx.sab: this world has no shared SAB (render-only world)");
        }
        return s;
      },
    };
    const comp = Object.assign(ref, create(ref, ctx)) as T;
    nextCompRef = { [$CompRef]: indexCompRef++ };
    return comp;
  };
}

export interface ComponentSab {
  readonly isProducer: boolean;
  readBank(): number;
  writeBank(): number;
  banks(name: string): NestedArray<Float64ArrayConstructor>[];
  pushOp(encode: (payload: Float64Array, slot: number) => number): void;
}

const componentSabFactories = new WeakMap<World, () => ComponentSab>();
const componentSabInstances = new WeakMap<World, ComponentSab>();

export function setComponentSabFactory(world: World, factory: () => ComponentSab): void {
  componentSabFactories.set(world, factory);
}

export function getComponentSab(world: World): ComponentSab | undefined {
  const existing = componentSabInstances.get(world);
  if (existing !== undefined) return existing;
  const factory = componentSabFactories.get(world);
  if (factory === undefined) return undefined;
  const sab = factory();
  componentSabInstances.set(world, sab);
  return sab;
}
