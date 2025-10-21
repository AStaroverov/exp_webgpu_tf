import { mean } from "../../lib/math";

/**
 * V-trace diagnostics: compute stats & health flags.
 * Input arrays can be number[] or Float32Array (same length each).
 */
export type VTraceDiagnostics = {
    advantages: {
        mean: number;
        std: number;
        p05: number;
        p95: number;
        abs95: number;
        flags: {
            advStdHigh: boolean;       // std > 2.5
            advTailsWide: boolean;     // |adv| 95% > 5
            advMeanShift: boolean;     // |mean| > 0.05
        };
    };
    tdErrors: {
        mean: number;
        std: number;
        p05: number;
        p95: number;
        flags: {
            tdMeanShift: boolean;      // |mean| > 0.05 * std
            tdHeavyTails: boolean;     // |p95| > 3 * std
        };
        corrWithAdvantages: number;  // Pearson
    };
    returns: {
        mean: number;
        std: number;
        min: number;
        max: number;
    };
    values: {
        mean: number;
        std: number;
    };
    fit: {
        explainedVariance: number;   // 1 - Var(R - V) / Var(R)
        scaleMismatch: boolean;      // std(V) deviates > 2x from std(R)
    };
};

export function analyzeVTrace(
    advantages: ArrayLike<number>,
    tdErrors: ArrayLike<number>,
    returns: ArrayLike<number>,
    values: ArrayLike<number>
): VTraceDiagnostics {
    const adv = toArray(advantages);
    const tde = toArray(tdErrors);
    const ret = toArray(returns);
    const val = toArray(values);

    // --- helpers ---
    const advMean = mean(adv);
    const advStd = stddev(adv, advMean);
    const advP05 = quantile(adv, 0.05);
    const advP95 = quantile(adv, 0.95);
    const advAbs95 = Math.max(Math.abs(advP05), Math.abs(advP95));

    const tdeMean = mean(tde);
    const tdeStd = stddev(tde, tdeMean);
    const tdeP05 = quantile(tde, 0.05);
    const tdeP95 = quantile(tde, 0.95);

    const retMean = mean(ret);
    const retStd = stddev(ret, retMean);
    const retMin = Math.min(...ret);
    const retMax = Math.max(...ret);

    const valMean = mean(val);
    const valStd = stddev(val, valMean);

    const ev = explainedVariance(ret, val, retMean);

    // correlations & flags
    const corrAdvTde = pearson(adv, tde);

    const flags = {
        advStdHigh: advStd > 2.5,
        advTailsWide: advAbs95 > 5.0,
        advMeanShift: Math.abs(advMean) > 0.05,

        tdMeanShift: Math.abs(tdeMean) > 0.05 * (tdeStd || 1),
        tdHeavyTails: Math.max(Math.abs(tdeP05), Math.abs(tdeP95)) > 3 * (tdeStd || 1),

        scaleMismatch: (valStd > 0 && retStd > 0) ? (valStd / retStd > 2 || retStd / valStd > 2) : false,
    };

    return {
        advantages: {
            mean: advMean,
            std: advStd,
            p05: advP05,
            p95: advP95,
            abs95: advAbs95,
            flags: {
                advStdHigh: flags.advStdHigh,
                advTailsWide: flags.advTailsWide,
                advMeanShift: flags.advMeanShift,
            },
        },
        tdErrors: {
            mean: tdeMean,
            std: tdeStd,
            p05: tdeP05,
            p95: tdeP95,
            flags: {
                tdMeanShift: flags.tdMeanShift,
                tdHeavyTails: flags.tdHeavyTails,
            },
            corrWithAdvantages: corrAdvTde,
        },
        returns: {
            mean: retMean,
            std: retStd,
            min: retMin,
            max: retMax,
        },
        values: {
            mean: valMean,
            std: valStd,
        },
        fit: {
            explainedVariance: ev,
            scaleMismatch: flags.scaleMismatch,
        },
    };
}

// ----------------- utils -----------------

function toArray(a: ArrayLike<number>): number[] {
    return Array.prototype.slice.call(a) as number[];
}

function variance(x: number[], m?: number): number {
    if (x.length === 0) return 0;
    const mu = m ?? mean(x);
    let s = 0;
    for (let i = 0; i < x.length; i++) {
        const d = x[i] - mu;
        s += d * d;
    }
    return s / x.length; // population variance (стабильнее для мониторинга)
}

function stddev(x: number[], m?: number): number {
    const v = variance(x, m);
    return v > 0 ? Math.sqrt(v) : 0;
}

function explainedVariance(y: number[], yhat: number[], yMean?: number): number {
    // EV = 1 - Var(y - yhat) / Var(y)
    const mu = yMean ?? mean(y);
    const diff = new Array(y.length);
    for (let i = 0; i < y.length; i++) diff[i] = y[i] - yhat[i];
    const varY = variance(y, mu);
    const varDiff = variance(diff, mean(diff));
    if (varY <= 1e-12) return 0; // неопределён масштаб
    return 1 - varDiff / varY;
}

function quantile(x: number[], q: number): number {
    if (x.length === 0) return 0;
    const xs = [...x].sort((a, b) => a - b);
    const idx = (xs.length - 1) * q;
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    if (lo === hi) return xs[lo];
    const w = idx - lo;
    return xs[lo] * (1 - w) + xs[hi] * w;
}

function pearson(x: number[], y: number[]): number {
    const n = x.length;
    if (n === 0) return 0;
    const mx = mean(x), my = mean(y);
    let num = 0, vx = 0, vy = 0;
    for (let i = 0; i < n; i++) {
        const dx = x[i] - mx;
        const dy = y[i] - my;
        num += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    const den = Math.sqrt(vx * vy);
    return den > 0 ? (num / den) : 0;
}

/**
{
    "advantages": {
        "mean": -1.1670713360851355e-10,
        "std": 0.9999999847935098,
        "p05": -1.5740155935287476,
        "p95": 1.567543601989746,
        "abs95": 1.5740155935287476,
        "flags": {
            "advStdHigh": false,
            "advTailsWide": false,
            "advMeanShift": false
        }
    },
    "tdErrors": {
        "mean": -0.0011501128797765155,
        "std": 0.3115187817083626,
        "p05": -0.44471889436244966,
        "p95": 0.42907680869102466,
        "flags": {
            "tdMeanShift": false,
            "tdHeavyTails": false
        },
        "corrWithAdvantages": 0.46058106546133876
    },
    "returns": {
        "mean": -0.7557363912044976,
        "std": 1.0969757593382714,
        "min": -5.877536773681641,
        "max": 3.3460586071014404
    },
    "values": {
        "mean": -0.8006618901329823,
        "std": 0.879443954956246
    },
    "fit": {
        "explainedVariance": 0.618930671051888,
        "scaleMismatch": false
    }
}
*/