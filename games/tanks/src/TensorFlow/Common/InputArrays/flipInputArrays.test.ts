import { strict as assert } from 'assert';
import { flipInputArrays } from './flipInputArrays';
import { ALLY_FEATURES_DIM } from '../../Models/Create.ts';

const sampleInput = {
    // controller: [shoot, move, rotate, dirX, dirY]
    controllerFeatures: new Float32Array([1, 0, 0, 1, 1]),

    // battle: [log(width), log(height), …]
    battleFeatures: new Float32Array([Math.log1p(10), Math.log1p(20), 0, 0, 0, 0]),

    // tank: [hp, Δx, Δy, rot, speedX, speedY, turretΔx, turretΔy, colliderR]
    tankFeatures: new Float32Array([1, 1, 2, 1, 3, 4, 5, 6, 1]),

    // один враг в первом слоте
    enemiesFeatures: new Float32Array([
        0.5, 1, 2, 3, 4, 5, 6, 1,   // slot-0
        0, 0, 0, 0, 0, 0, 0, 0,   // slot-1 (пустой)
    ]),
    enemiesMask: new Float32Array([1, 0]),

    // союзников нет — достаточно пустого массива нужной длины
    alliesFeatures: new Float32Array(ALLY_FEATURES_DIM * 2),
    alliesMask: new Float32Array([0, 0]),

    // одна пуля
    bulletsFeatures: new Float32Array([
        1, 2, 3, 4,
        0, 0, 0, 0,
    ]),
    bulletsMask: new Float32Array([1, 0]),
} as const;

// ────────────────────────────────────
// Помощник: проверка dx/dy для tank & enemy
// ────────────────────────────────────
function tankXY(out: ReturnType<typeof flipInputArrays>) {
    return { dx: out.tankFeatures[1], dy: out.tankFeatures[2] };
}

function enemyXY(out: ReturnType<typeof flipInputArrays>) {
    return { dx: out.enemiesFeatures[1], dy: out.enemiesFeatures[2] };
}

const EPS = 0.0001;
// ────────────────────────────────────
// Тесты
// ────────────────────────────────────
(function testFlipX() {
    const fx = flipInputArrays(sampleInput, 'x');

    // Δy и все «Y»-компоненты должны сменить знак
    assert.equal(tankXY(fx).dy, -2);
    assert.equal(enemyXY(fx).dy, -2);
    assert.equal(fx.bulletsFeatures[1], -2);      // пуля: Δy
    assert.equal(fx.tankFeatures[5], -4);         // speedY

    // угол тоже инвертируется
    assert.equal(fx.tankFeatures[3], -1);
})();

(function testFlipY() {
    const fy = flipInputArrays(sampleInput, 'y');

    // Δx меняет знак
    assert.equal(tankXY(fy).dx, -1);
    assert.equal(enemyXY(fy).dx, -1);
    assert.equal(fy.bulletsFeatures[0], -1);      // пуля: Δx
    assert.equal(fy.tankFeatures[4], -3);         // speedX

    // угол: π - old
    if (fy.tankFeatures[3] - (Math.PI - 1) > EPS) {
        assert.equal(fy.tankFeatures[3], (Math.PI - 1));
    }
})();

(function testFlipXY() {
    const fxy = flipInputArrays(sampleInput, 'xy');

    // X ↔ Y
    assert.deepEqual(tankXY(fxy), { dx: 2, dy: 1 });
    assert.deepEqual(enemyXY(fxy), { dx: 2, dy: 1 });

    // ширина ↔ высота
    if (fxy.battleFeatures[0] - Math.log1p(20) > EPS)
        assert.equal(fxy.battleFeatures[0], Math.log1p(20));
    if (fxy.battleFeatures[1] - Math.log1p(10) > EPS)
        assert.equal(fxy.battleFeatures[1], Math.log1p(10));

    // угол: π/2 – old
    if (fxy.tankFeatures[3] - (Math.PI / 2 - 1) > EPS)
        assert.equal(fxy.tankFeatures[3], Math.PI / 2 - 1);
})();

console.log('✅ flipInputArrays tests passed');
