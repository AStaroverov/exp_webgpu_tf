import * as tf from '@tensorflow/tfjs';

type MultiOUOpts = {
    K?: number;           // число фильтров (шкал), 6..12 обычно достаточно
    tauMin?: number;      // минимальная "длина" шкалы в шагах
    tauMax?: number;      // максимальная "длина" шкалы в шагах
};

export class ColoredNoiseApprox {
    private headDims: number[];
    private K: number;
    private a: tf.Tensor;         // коэффициенты AR(1): x <- a x + b z
    private b: tf.Tensor;         // чтобы var(x_k)=1: b = sqrt(1 - a^2)
    private w: tf.Tensor;         // веса по шкалам, ||w||_2 = 1
    private states: tf.Tensor[];  // массив [K, headDim] для каждой головы
    private disposed = false;

    constructor(headDims: number[], beta = 1, opts: MultiOUOpts = {}) {
        const {
            K = 8,
            tauMin = 4,
            tauMax = 64,
        } = opts;
        this.headDims = headDims;
        this.K = K;

        const { a, b, w } = tf.tidy(() => {
            const logMin = Math.log(tauMin);
            const logMax = Math.log(tauMax);
            const logs = tf.linspace(logMin, logMax, K);
            const taus = tf.exp(logs);
            const aVec = tf.exp(tf.neg(tf.div(tf.onesLike(taus), taus)));
            const bVec = tf.sqrt(tf.maximum(tf.sub(1, tf.square(aVec)), 1e-8));

            const tilt = (beta - 1) / 2;
            const wRaw = tf.pow(taus, tf.scalar(-tilt));
            const wVec = tf.div(wRaw, tf.sqrt(tf.sum(tf.square(wRaw))));

            return {
                a: aVec.reshape([K, 1]) as tf.Tensor2D,
                b: bVec.reshape([K, 1]) as tf.Tensor2D,
                w: wVec.reshape([K, 1]) as tf.Tensor2D
            };
        });

        this.a = a;
        this.b = b;
        this.w = w;
        this.states = headDims.map(dim => tf.zeros([K, dim]) as tf.Tensor2D);
    }

    sample(): tf.Tensor[] {
        if (this.disposed) throw new Error('ColoredNoise: disposed');

        const {K, a, b, w} = this;
        const states: tf.Tensor[] = [];
        const outputs: tf.Tensor[] = [];

        for (let headIdx = 0; headIdx < this.headDims.length; headIdx++) {
            const D = this.headDims[headIdx];
            const { state, out } = tf.tidy(() => {
                const z = tf.randomNormal([K, D]);
                const s = a.mul(this.states[headIdx]).add(b.mul(z)); // [K,D]
                const y = s.mul(w).sum(0);

                return { state: s, out: y };
            });

            states.push(state);
            outputs.push(out);
        }

        this.states.forEach(s => s.dispose());
        this.states = states;
        return outputs;
    }

    dispose() {
        if (this.disposed) return;
        this.disposed = true;
        this.states.forEach(s => s.dispose());
        this.a.dispose();
        this.b.dispose();
        this.w.dispose();
    }
}
