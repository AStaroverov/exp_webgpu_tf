export const max = Math.max;
export const min = Math.min;
export const abs = Math.abs;
export const log = Math.log;
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

export function smoothstep(a: number, b: number, x: number): number {
    const t = max(0, min(1, (x - a) / (b - a)));
    return t * t * (3 - 2 * t);
}

export function centerStep(a: number, b: number, x: number): number {
    let t = (x - a) / (b - a);
    t = 2 * t - 1;
    return Math.max(0, 1 - t * t);
}

export function lerp(a: number, b: number, x: number): number {
    return a + (b - a) * x;
}

/**
 * Функция, которая применяет логарифмическое преобразование с сохранением знака.
 * @param {number} x - исходное значение.
 * @param {number} base - параметр, регулирующий степень сжатия (например, 10).
 * @returns {number} Преобразованное значение.
 */
export function signedLog(x: number, base: number): number {
    // Формула: sign(x) * log(1 + |x|) / log(1 + base)
    return sign(x) * log(1 + abs(x)) / log(1 + base);
}

/**
 * Функция для линейного масштабирования массива значений в заданный диапазон.
 * @param {number[]} arr - массив чисел для масштабирования.
 * @param {number} targetMin - минимальное значение целевого диапазона.
 * @param {number} targetMax - максимальное значение целевого диапазона.
 * @returns {number[]} Масштабированный массив.
 */
export function linearScale(arr: number[], targetMin: number, targetMax: number): number[] {
    const minVal = min(...arr);
    const maxVal = max(...arr);
    // Если все значения равны, вернем массив с targetMin (или можно иначе обработать)
    if (maxVal === minVal) {
        return arr.map(() => targetMin);
    }
    return arr.map(val =>
        ((val - minVal) / (maxVal - minVal)) * (targetMax - targetMin) + targetMin,
    );
}