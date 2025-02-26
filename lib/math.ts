export const max = Math.max;
export const min = Math.min;
export const abs = Math.abs;
export const sign = Math.sign;
export const floor = Math.floor;
export const ceil = Math.ceil;
export const round = Math.round;
export const sin = Math.sin;
export const cos = Math.cos;
export const trunc = Math.trunc;
export const sqrt = Math.sqrt;
export const atan2 = Math.atan2;
export const hypot = Math.hypot;

export function ufloor(n: number): number {
    return sign(n) * floor(abs(n));
}

export function uceil(n: number): number {
    return sign(n) * ceil(abs(n));
}

export function uround(n: number): number {
    return sign(n) * round(abs(n));
}

export function dist2(x1: number, y1: number, x2: number, y2: number): number {
    return hypot(x2 - x1, y2 - y1);
}

export function smoothstep(edge0: number, edge1: number, x: number): number {
    const t = max(0, min(1, (x - edge0) / (edge1 - edge0)));
    return t * t * (3 - 2 * t);
}