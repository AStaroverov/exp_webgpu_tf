import * as tf from '@tensorflow/tfjs';

type MultiOUOpts = {
    K?: number;           // число фильтров (шкал), 6..12 обычно достаточно
    tauMin?: number;      // минимальная "длина" шкалы в шагах
    tauMax?: number;      // максимальная "длина" шкалы в шагах
};

export class ColoredNoiseApprox {
    private actionDim: number;
    private K: number;
    private a: tf.Tensor1D;      // [K] коэффициенты AR(1): x <- a x + b z
    private b: tf.Tensor1D;      // [K] чтобы var(x_k)=1: b = sqrt(1 - a^2)
    private w: tf.Tensor1D;      // [K] веса по шкалам, ||w||_2 = 1
    private state: tf.Tensor2D;  // [K, D]
    private disposed = false;

    constructor(actionDim: number, beta = 1, opts: MultiOUOpts = {}) {
        const {
            K = 8,
            tauMin = 4,
            tauMax = 64,
        } = opts;
        this.actionDim = actionDim;
        this.K = K;

        const { a, b, w } = tf.tidy(() => {
            // Лог-равномерные τ_k в [tauMin, tauMax]
            const logMin = Math.log(tauMin);
            const logMax = Math.log(tauMax);
            const logs = tf.linspace(logMin, logMax, K);
            const taus = tf.exp(logs);                      // [K]
            // a_k = exp(-1/τ_k)
            const a = tf.exp(tf.neg(tf.div(tf.onesLike(taus), taus))); // [K]
            // b_k = sqrt(1 - a_k^2) => var(x_k)=1
            const b = tf.sqrt(tf.maximum(tf.sub(1, tf.square(a)), 1e-8));

            // Веса по шкалам: для β≈1 достаточно одинаковых весов на лог-шкалах.
            // Для общего β — слегка «наклоняем» спектр: w ~ τ^{- (β-1)/2}, потом L2-норм.
            const tilt = (beta - 1) / 2;
            const wRaw = tf.pow(taus, tf.scalar(-tilt));
            const w = tf.div(wRaw, tf.sqrt(tf.sum(tf.square(wRaw)))); // ||w||2=1

            return { a, b, w };
        });

        this.a = a as tf.Tensor1D;
        this.b = b as tf.Tensor1D;
        this.w = w as tf.Tensor1D;

        // Инициализация состояния K фильтров на D каналов (ноль — ок, стац. режим быстро достигается)
        this.state = tf.zeros([K, actionDim]) as tf.Tensor2D;
    }

    /** Возвращает вектор [D] ~ N(0,1) с 1/f^beta спектром по времени. */
    sample(): tf.Tensor1D {
        if (this.disposed) throw new Error('ColoredNoise: disposed');

        const { newState, out } = tf.tidy(() => {
            const K = this.K;
            const D = this.actionDim;
            const a2d = this.a.reshape([K, 1]);      // [K,1]
            const b2d = this.b.reshape([K, 1]);      // [K,1]
            const w2d = this.w.reshape([K, 1]);      // [K,1]

            const z = tf.randomNormal([K, D]);       // независимые возбуждения
            const nextState = tf.add(tf.mul(a2d, this.state), tf.mul(b2d, z)) as tf.Tensor2D; // [K,D]
            const y = tf.sum(tf.mul(nextState, w2d), 0) as tf.Tensor1D; // взвешенная сумма по шкалам -> [D]

            return { newState: nextState, out: y };
        });

        this.state.dispose();
        this.state = newState;
        return out;
    }

    dispose() {
        if (this.disposed) return;
        this.disposed = true;
        this.state.dispose();
        this.a.dispose();
        this.b.dispose();
        this.w.dispose();
    }
}
