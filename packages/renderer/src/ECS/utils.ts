import { addComponent } from "bitecs";
import type { World } from "bitecs";

const $CompRef = Symbol("CompRef");
let indexCompRef = 0;
let nextCompRef: any = { [$CompRef]: indexCompRef++ };

export function component<T>(_comp: T): T {
  const comp = Object.assign(nextCompRef, _comp);
  nextCompRef = { [$CompRef]: indexCompRef++ };
  return comp;
}

type ReactiveSetter = (eid: number, ...args: any[]) => any;
export type Obs = <F extends ReactiveSetter>(setter: F) => F;

export function defineComponent<T extends object>(create: (ref: object, obs: Obs) => T) {
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
    const comp = Object.assign(ref, create(ref, localObs)) as T;
    nextCompRef = { [$CompRef]: indexCompRef++ };
    return comp;
  };
}
