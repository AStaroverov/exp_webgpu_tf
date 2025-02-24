import { addComponent } from 'bitecs';
import { DI } from '../games/tanks/src/DI';

export function uniq<T>(arr: T[]): T[] {
    return Array.from(new Set(arr));
}

export function uniqBy<T>(arr: T[], predicate: (v: T) => unknown): T[] {
    return Array.from(arr.reduce((map, item) => {
        const key = (item === null || item === undefined)
            ? item
            : predicate(item);

        map.has(key) || map.set(key, item);

        return map;
    }, new Map()), ([, v]) => v);
}


export function isNil<T>(v: T | null | undefined): v is null | undefined {
    return v === null || v === undefined;
}

export function notNil<T>(v: T | null | undefined): v is T {
    return v !== null && v !== undefined;
}

export class TypedArray {
    public static f64 = (length: number) => new Float64Array(length);
    public static f32 = (length: number) => new Float32Array(length);
    public static u32 = (length: number) => new Uint32Array(length);
    public static i32 = (length: number) => new Int32Array(length);
    public static i8 = (length: number) => new Int8Array(length);
}

type ArrayLikeConstructor =
    Float64ArrayConstructor |
    Float32ArrayConstructor |
    Uint32ArrayConstructor |
    Int32ArrayConstructor;

export class NestedArray<T extends ArrayLikeConstructor> {
    public buffer: T['prototype'];
    public bufferLength: number;

    constructor(kind: T, public batchLength: number, public batchCount: number, seed?: ArrayLike<number>) {
        this.buffer = new kind(batchLength * batchCount);
        this.bufferLength = this.buffer.length;

        seed && this.buffer.set(seed);
    }

    public static f64 = (batchLength: number, batchCount: number, seed?: ArrayLike<number>) => new NestedArray(Float64Array, batchLength, batchCount, seed);

    public static f32 = (batchLength: number, batchCount: number, seed?: ArrayLike<number>) => new NestedArray(Float32Array, batchLength, batchCount, seed);

    public static f16 = (batchLength: number, batchCount: number, seed?: ArrayLike<number>) => new NestedArray(Float32Array, batchLength, batchCount, seed);

    public static u32 = (batchLength: number, batchCount: number, seed?: ArrayLike<number>) => new NestedArray(Uint32Array, batchLength, batchCount, seed);

    public static i32 = (batchLength: number, batchCount: number, seed?: ArrayLike<number>) => new NestedArray(Int32Array, batchLength, batchCount, seed);

    destroy() {
        // @ts-ignore
        this.buffer = null;
        this.bufferLength = 0;
    }

    public get(batchIndex: number, index: number): number {
        return this.buffer[batchIndex * this.batchLength + index];
    }

    public set(batchIndex: number, index: number, value: number) {
        this.buffer[batchIndex * this.batchLength + index] = value;
    }

    public fill(value: number) {
        this.buffer.fill(value);
    }

    public setBatch(batchIndex: number, values: ArrayLike<number>) {
        this.buffer.set(values, batchIndex * this.batchLength);
    }

    public getBatche(batchStart: number): T['prototype'] {
        return this.buffer.subarray(batchStart * this.batchLength, (batchStart + 1) * this.batchLength);
    }
}

const $CompRef = Symbol('CompRef');
let indexCompRef = 0;
let nextCompRef: any = { [$CompRef]: indexCompRef++ };

export function component<T>(_comp: T): T {
    const comp = Object.assign(nextCompRef, _comp);
    nextCompRef = { [$CompRef]: indexCompRef++ };
    return comp;
}

export function obs<T extends (eid: number, ...args: A) => void, A extends any[]>(setter: T): T {
    const setData = { component: nextCompRef, data: null };
    return ((eid: number, ...args: A) => {
        const r = setter(eid, ...args);
        addComponent(DI.world, eid, setData);
        return r;
    }) as T;
}
