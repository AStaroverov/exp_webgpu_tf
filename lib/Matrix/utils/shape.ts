const { sign, sqrt } = Math;

export type TSize = { w: number; h: number };
export const newSize = (w: number, h?: number): TSize => ({ w, h: h ?? w });

export const Size = {
    create: newSize,
    set: (t: TSize, s: TSize): TSize => {
        t.w = s.w;
        t.h = s.h;
        return t;
    },
    toVector: (s: TSize) => Vector.create(s.w, s.h),
    fromVector: (v: TVector) => Size.create(v.x, v.y),

    ZERO: newSize(0, 0),
};

export type Point = { x: number; y: number };
export const newPoint = (x: number, y: number): Point => ({ x, y });

// VECTOR
export type TVector = { x: number; y: number };

export const newVector = (x: number, y: number): TVector => ({ x, y });

export const extractVector = (o: object & TVector): TVector => newVector(o.x, o.y);

export const setVector = (t: TVector, s: TVector): TVector => {
    t.x = s.x;
    t.y = s.y;
    return t;
};
export const copyVector = (v: TVector): TVector => newVector(v.x, v.y);

export const mapVector = (t: TVector, map: (v: number) => number): TVector =>
    newVector(map(t.x), map(t.y));

export const sumVector = (f: TVector, ...vs: TVector[]): TVector => {
    return vs.reduce((sum, v) => {
        sum.x += v.x;
        sum.y += v.y;
        return sum;
    }, copyVector(f));
};

export const mulVector = (v: TVector, k: number | TVector): TVector =>
    typeof k === 'number' ? newVector(v.x * k, v.y * k) : newVector(v.x * k.x, v.y * k.y);

export const negateVector = (v: TVector): TVector => mulVector(v, -1);

export const widthVector = (a: TVector): number => sqrt(a.x ** 2 + a.y ** 2);

export const distanceVector = (a: TVector, b: TVector): number =>
    widthVector(newVector(b.x - a.x, b.y - a.y));

export const normalize = (a: TVector): TVector => {
    const width = widthVector(a);
    return newVector(a.x / width, a.y / width);
};

export const isEqualVectors = (a: TVector, b: TVector): boolean => a.x === b.x && a.y === b.y;

export const hasEqualDirection = (a: TVector, b: TVector): boolean =>
    isEqualVectors(mapVector(a, sign), mapVector(b, sign));

export const isOneWayDirection = (v: TVector): boolean => {
    return (v.x === 0 && v.y !== 0) || (v.x !== 0 && v.y === 0);
};

export const toOneWayVectors = (a: TVector): TVector[] => {
    if (isEqualVectors(a, zeroVector)) return [zeroVector];

    const result: TVector[] = [];

    a.x !== 0 && result.push(newVector(a.x, 0));
    a.y !== 0 && result.push(newVector(0, a.y));

    return result;
};

export const toStringVector = (v: TVector): string => `Vector{${ v.x },${ v.y }`;

export const zeroVector = newVector(0, 0);

export const Vector = {
    create: newVector,
    extract: extractVector,
    set: setVector,
    copy: copyVector,
    map: mapVector,
    sum: sumVector,
    mul: mulVector,
    negate: negateVector,
    width: widthVector,
    distance: distanceVector,
    normalize: normalize,
    isEqual: isEqualVectors,
    hasEqualDirection: hasEqualDirection,
    isOneWayDirection: isOneWayDirection,
    toOneWayVectors: toOneWayVectors,
    toString: toStringVector,
    ZERO: zeroVector,
    UP: newVector(0, 1),
    DOWN: newVector(0, -1),
    LEFT: newVector(-1, 0),
    RIGHT: newVector(1, 0),
};
