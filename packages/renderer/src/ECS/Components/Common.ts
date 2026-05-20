import { delegate } from '../../delegate.ts';
import { NestedArray, TypedArray } from '../../utils.ts';
import { addComponent, World } from 'bitecs';
import { defineComponent } from '../utils.ts';

export const createThinnessComponent = defineComponent((Thinness, obs) => {
    const value = TypedArray.f64(delegate.defaultSize);
    return {
        value,
        addComponent(world: World, eid: number, v = 0) {
            addComponent(world, eid, Thinness);
            value[eid] = v;
        },
        set$: obs((eid: number, v: number) => {
            value[eid] = v;
        }),
    };
});

export const createRoundnessComponent = defineComponent((Roundness, obs) => {
    const value = new Float64Array(delegate.defaultSize);
    return {
        value,
        addComponent(world: World, eid: number, v = 0) {
            addComponent(world, eid, Roundness);
            value[eid] = v;
        },
        set$: obs((eid: number, v: number) => {
            value[eid] = v;
        }),
    };
});

export type TColor = [number, number, number, number] | Float32Array;

export const createColorComponent = defineComponent((Color, obs) => {
    const rgba = NestedArray.f64(4, delegate.defaultSize);

    return {
        rgba,
        addComponent(world: World, eid: number, r = 0, g = 0, b = 0, a = 1) {
            addComponent(world, eid, Color);
            rgba.set(eid, 0, r).set(eid, 1, g).set(eid, 2, b).set(eid, 3, a);
        },
        set$: obs((eid: number, r: number, g: number, b: number, a: number) => {
            rgba.set(eid, 0, r).set(eid, 1, g).set(eid, 2, b).set(eid, 3, a);
        }),

        getArray: (eid: number) => rgba.getBatch(eid),
        getR: (eid: number) => rgba.get(eid, 0),
        getG: (eid: number) => rgba.get(eid, 1),
        getB: (eid: number) => rgba.get(eid, 2),
        getA: (eid: number) => rgba.get(eid, 3),

        setR$: obs((eid: number, v: number) => rgba.set(eid, 0, v)),
        setG$: obs((eid: number, v: number) => rgba.set(eid, 1, v)),
        setB$: obs((eid: number, v: number) => rgba.set(eid, 2, v)),
        setA$: obs((eid: number, v: number) => rgba.set(eid, 3, v)),

        applyColorToArray<T extends TColor>(eid: number, color: T): T {
            color[0] = rgba.get(eid, 0);
            color[1] = rgba.get(eid, 1);
            color[2] = rgba.get(eid, 2);
            color[3] = rgba.get(eid, 3);
            return color;
        },
    };
});
