import { computeVTrace } from './train';

function approx(a: number, b: number, eps = 1e-6) {
    console.assert(Math.abs(a - b) < eps, `|${ a }-${ b }|>=${ eps }`);
}

function arrayApprox(a: Float32Array, b: Float32Array, eps = 1e-6) {
    console.assert(a.length === b.length, 'length mismatch');
    for (let i = 0; i < a.length; ++i) approx(a[i], b[i], eps);
}

(function testAllDone() {
    const n = 5;
    const vals = Float32Array.from([1, 2, 3, 4, 5]);
    const dones = Float32Array.from([1, 1, 1, 1, 1]);
    const zeros = new Float32Array(n);
    const ones = Float32Array.from({ length: n }, () => 1);

    const { vTraces } = computeVTrace(zeros, dones, vals, ones, 0.99, 1e9, 1e9);

    arrayApprox(vTraces, vals);
    console.log('[V-Trace] ✓ all‑done test passed');
})();

(function testGamma0() {
    const rewards = Float32Array.from([10, 20, 30]);
    const dones = new Float32Array([0, 0, 0]);
    const values = new Float32Array([0, 0, 0]);
    const rhos = new Float32Array([1, 1, 1]);

    const { vTraces } = computeVTrace(rewards, dones, values, rhos, 0, 1e9, 1e9);

    arrayApprox(vTraces, rewards);
    console.log('[V-Trace] ✓ gamma=0 test passed');
})();

(function testRhoZero() {
    const n = 6;
    const vals = Float32Array.from({ length: n }, (_, i) => i);
    const dones = new Float32Array(n).fill(0);
    const zeros = new Float32Array(n);

    const { vTraces } = computeVTrace(
        zeros, dones, vals, zeros, 0.99, 1.0, 1.0,
    );
    arrayApprox(vTraces, vals);
    console.log('[V-Trace] ✓ rho=0 test passed');
})();

(function complextTest() {
    const N = 30;

    // 1) rewards = [1, 2, …, 30]
    const rewards = Float32Array.from(Array.from({ length: N }, (_, i) => i + 1));

    // 2) values = rewards + noise in [-0.1, +0.1]
    const values = Float32Array.from(
        rewards.map(r => r + (Math.random() * 0.2 - 0.1)),
    );

    // 3) терминалы на шагах 9, 19, 29
    const dones = Float32Array.from(
        Array.from({ length: N }, (_, i) => [9, 19, 29].includes(i) ? 1 : 0),
    );

    // 4) ρ=1, γ=1
    const rhos = new Float32Array(N).fill(1);
    const gamma = 1;
    const clipRho = 1e6;
    const clipC = 1e6;

    const { vTraces } = computeVTrace(rewards, dones, values, rhos, gamma, clipRho, clipC);

    // 5) эталон: сумма rewards до терминации (values не влияют)
    const expected = new Float32Array(N);
    for (let t = 0; t < N; t++) {
        let sum = 0;
        for (let k = t; k < N; k++) {
            sum += rewards[k];
            if (dones[k] === 1) break;
        }
        expected[t] = sum;
    }

    arrayApprox(vTraces, expected, 1);
    console.log('[V-Trace] ✓ full V-trace passed');
})();