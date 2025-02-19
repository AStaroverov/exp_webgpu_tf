import { observe, World } from 'bitecs';
import { ObservableHook } from 'bitecs/dist/core/Query';

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

    public setBatch(batchIndex: number, values: ArrayLike<number>) {
        this.buffer.set(values, batchIndex * this.batchLength);
    }

    public getBatche(batchStart: number): T['prototype'] {
        return this.buffer.subarray(batchStart * this.batchLength, (batchStart + 1) * this.batchLength);
    }
}

export function createChangedDetector(world: World, hooks: ObservableHook[]) {
    const value = new Set<number>();
    const stops = hooks.map((hook) => observe(world, hook, (eid) => value.add(eid)));
    const stop = () => stops.forEach((s) => s());
    const hasChanges = () => value.size > 0;
    const has = (eid: number) => value.has(eid);
    const clear = () => value.clear();
    const destroy = () => {
        stop();
        value.clear();
    };

    return { has, hasChanges, clear, destroy };
}