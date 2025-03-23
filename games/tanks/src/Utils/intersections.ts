import { sqrt } from '../../../../lib/math.ts';

export function hasIntersectionVectorAndCircle(
    x1: number,
    y1: number,
    dx: number,
    dy: number,
    cx: number,
    cy: number,
    radius: number,
): boolean {
    const fx = x1 - cx;
    const fy = y1 - cy;

    const a = dx * dx + dy * dy;
    const b = 2 * (fx * dx + fy * dy);
    const c = fx * fx + fy * fy - radius * radius;

    const discriminant = b * b - 4 * a * c;

    if (discriminant < 0) {
        // Нет пересечения
        return false;
    }

    // Ищем точки пересечения по параметру t (0 <= t <= 1)
    const sqrtDiscriminant = sqrt(discriminant);
    const t1 = (-b - sqrtDiscriminant) / (2 * a);
    const t2 = (-b + sqrtDiscriminant) / (2 * a);

    return (t1 >= 0 && t1 <= 1) || (t2 >= 0 && t2 <= 1);
}