import { addComponent } from "bitecs";
import type { World } from "bitecs";

const $CompRef = Symbol("CompRef");
let indexCompRef = 0;
let nextCompRef: any = { [$CompRef]: indexCompRef++ };

type ReactiveSetter = (eid: number, ...args: any[]) => any;

export type Obs = <F extends ReactiveSetter>(setter: F) => F;

export type ComponentContext = { obs: Obs; world: World };

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
        const result = setter(eid, ...args);
        addComponent(world, eid, setData);
        return result;
      }) as F;
    };
    const comp = Object.assign(ref, create(ref, { obs: localObs, world })) as T;
    nextCompRef = { [$CompRef]: indexCompRef++ };
    return comp;
  };
}
