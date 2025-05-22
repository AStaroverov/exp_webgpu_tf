export const PI = Math.PI;
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
export const acos = Math.acos;
export const trunc = Math.trunc;
export const sqrt = Math.sqrt;
export const atan2 = Math.atan2;
export const hypot = Math.hypot;
export const tanh = Math.tanh;
export const pow = Math.pow;

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
    return max(0, 1 - t * t);
}

export function lerp(a: number, b: number, x: number): number {
    return a + (b - a) * x;
}

export function mean(args: number[] | Float32Array | Float64Array): number {
    let sum = 0;
    for (let i = 0; i < args.length; i++) {
        sum += args[i];
    }
    return sum / args.length;
}

export function std(args: number[] | Float32Array | Float64Array, mean: number): number {
    let val = 0;
    for (let i = 0; i < args.length; i++) {
        const diff = args[i] - mean;
        val += diff * diff;
    }
    val /= args.length;
    return sqrt(val);
}

export function normalize<T extends number[] | Float32Array | Float64Array>(args: T): T {
    const m = mean(args);
    const s = std(args, m) + 1e-8;
    return args.map((v) => (v - m) / s) as T;
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

/**
 * Выполняет Winsorization (обрезание выбросов по заданным процентилям)
 *
 * @param advantages - Исходный массив преимуществ (может содержать отрицательные, положительные).
 * @param lowerPerc - Нижний процентиль (0..1). Например, 0.05 обрежет нижние 5%.
 * @param upperPerc - Верхний процентиль (0..1). Например, 0.95 обрежет верхние 5%.
 * @returns Новый массив, где выбросы обрезаны, а значения нормализованы.
 */
export function winsorize(
    advantages: number[],
    lowerPerc = 0.05,
    upperPerc = 0.95,
): number[] {
    if (advantages.length < 2) {
        // Если массив слишком короткий, вернём копию (или можно вернуть сам массив)
        return [...advantages];
    }

    // 1) Копируем и сортируем массив, чтобы найти пороговые значения
    const sorted = [...advantages].sort((a, b) => a - b);
    const n = sorted.length;

    // Индексы для процентилей
    const lowerIndex = Math.floor(n * lowerPerc);
    const upperIndex = Math.floor(n * upperPerc);

    // Извлекаем значения на границах
    const lowerVal = sorted[Math.max(0, lowerIndex)];                 // защита от выхода за диапазон
    const upperVal = sorted[Math.min(upperIndex, n - 1)];

    // 2) Winsorization (обрезаем значения, выходящие за эти границы)
    const winsorized = advantages.map(a => {
        if (a < lowerVal) return lowerVal;
        if (a > upperVal) return upperVal;
        return a;
    });

    return winsorized;
}