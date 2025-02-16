import { TSize, TVector, Vector } from './shape';

export type TRect = TVector &
    TSize & {
    mx: number;
    my: number;
};
export const create = (x: number, y: number, w: number, h: number): TRect => {
    return { x, y, w, h, mx: x + w, my: y + h };
};

export const map = (r: TRect, map: (v: number, i: number) => number): TRect => {
    return create(map(r.x, 0), map(r.y, 1), map(r.w, 2), map(r.h, 3));
};

export const add = (t: TRect, s: TRect): TRect => {
    return create(t.x + s.x, t.y + s.y, t.w + s.w, t.h + s.h);
};

export const zoomByCenter = (s: TRect, v: number | TVector): TRect => {
    const r = typeof v === 'number' ? Vector.create(v, v) : v;
    return create(s.x - r.x / 2, s.y - r.y / 2, s.w + r.x, s.h + r.y);
};

export const inside = (a: TRect, b: TRect): boolean => {
    return a.x >= b.x && a.y >= b.y && a.mx <= b.mx && a.my <= b.my;
};

export const pointInside = (r: TRect, p: TVector): boolean => {
    return r.x <= p.x && r.y <= p.y && r.mx >= p.x && r.my >= p.y;
};

export const notIntersect = (a: TRect, b: TRect): boolean => {
    return a.mx < b.x || a.x > b.mx || a.my < b.y || a.y > b.my;
};

export const intersect = (a: TRect, b: TRect): boolean => {
    return !notIntersect(a, b);
};

export const getCenter = (r: TRect): TVector => {
    return Vector.create(r.x + r.w / 2, r.y + r.h / 2);
};

export const getAllVertexes = (r: TRect): TVector[] => {
    return [
        Vector.create(r.x, r.y),
        Vector.create(r.x, r.my),
        Vector.create(r.mx, r.my),
        Vector.create(r.mx, r.y),
    ];
};

export const fromCenterAndSize = (c: TVector, s: TSize): TRect => {
    return create(c.x - s.w / 2, c.y - s.h / 2, c.x + s.w / 2, c.y + s.h / 2);
};

export const Rect = {
    create,
    map,
    add,
    zoomByCenter,
    inside,
    pointInside,
    intersect,
    notIntersect,

    getCenter,
    getAllVertexes,

    fromCenterAndSize,
};
