import { delegate } from '../../delegate.ts';
import { NestedArray, TypedArray } from '../../utils.ts';
import { addComponent, World } from 'bitecs';
import { component, obs } from '../utils.ts';

export const Thinness = component({
    value: TypedArray.f64(delegate.defaultSize),

    addComponent: (world: World, eid: number, value = 0) => {
        addComponent(world, eid, Thinness);
        Thinness.value[eid] = value;
    },
    set$: obs((eid: number, value: number) => {
        Thinness.value[eid] = value;
    }),
});

export const Roundness = component({
    value: new Float64Array(delegate.defaultSize),

    addComponent: (world: World, eid: number, value = 0) => {
        addComponent(world, eid, Roundness);
        Roundness.value[eid] = value;
    },
    set$: obs((eid: number, value: number) => {
        Roundness.value[eid] = value;
    }),
});

export type TColor = [number, number, number, number] | Float32Array;
export const Color = component({
    rgba: NestedArray.f64(4, delegate.defaultSize),

    addComponent(world: World, eid: number, r = 0, g = 0, b = 0, a = 1) {
        addComponent(world, eid, Color);
        Color.rgba.set(eid, 0, r).set(eid, 1, g).set(eid, 2, b).set(eid, 3, a);
    },
    set$: obs((eid: number, r: number, g: number, b: number, a: number) => {
        Color.rgba.set(eid, 0, r).set(eid, 1, g).set(eid, 2, b).set(eid, 3, a);
    }),

    getArray: (eid: number) => Color.rgba.getBatch(eid),
    getR: (eid: number) => Color.rgba.get(eid, 0),
    getG: (eid: number) => Color.rgba.get(eid, 1),
    getB: (eid: number) => Color.rgba.get(eid, 2),
    getA: (eid: number) => Color.rgba.get(eid, 3),

    setR$: obs((eid: number, v: number) => Color.rgba.set(eid, 0, v)),
    setG$: obs((eid: number, v: number) => Color.rgba.set(eid, 1, v)),
    setB$: obs((eid: number, v: number) => Color.rgba.set(eid, 2, v)),
    setA$: obs((eid: number, v: number) => Color.rgba.set(eid, 3, v)),

    applyColorToArray: <T extends TColor>(eid: number, color: T): T => {
        color[0] = Color.rgba.get(eid, 0);
        color[1] = Color.rgba.get(eid, 1);
        color[2] = Color.rgba.get(eid, 2);
        color[3] = Color.rgba.get(eid, 3);
        return color;
    },
});
