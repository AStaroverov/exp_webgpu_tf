import * as tf from '@tensorflow/tfjs';

/**
 * NoiseMatrix для gSDE (generalized State Dependent Exploration)
 * 
 * Управляет матрицей шума Θ размерности [latentDim, actionDim], которая:
 * - Периодически пересэмплируется (раз в noiseUpdateFrequency шагов)
 * - Используется для генерации state-dependent шума: ε(s) = φ(s) @ Θ
 * 
 * Варианты масштабирования:
 * - Вариант A: Θ ~ N(0, I), масштаб применяется после φ@Θ
 * - Вариант B: Θ ~ N(0, diag(exp(2*logStdBase))), без дополнительного масштаба
 */
export class NoiseMatrix {
    private Theta?: tf.Tensor2D; // [latentDim, actionDim]
    private step = 0;

    constructor(
        private latentDim: number,
        private actionDim: number,
        private noiseUpdateFrequency: number,
        public logStdBase: tf.Variable,  // shape [actionDim]
        private variantA = true          // true: вариант A, false: вариант B
    ) {
    }

    /**
     * Пересэмплирование матрицы шума Θ
     */
    resample() {
        this.Theta?.dispose();

        if (this.variantA) {
            // Вариант A: Θ ~ N(0, I)
            this.Theta = tf.randomNormal([this.latentDim, this.actionDim]) as tf.Tensor2D;
        } else {
            // Вариант B: Θ ~ N(0, diag(σ_base^2))
            this.Theta = tf.tidy(() => {
                const baseStd = tf.exp(this.logStdBase).reshape([1, this.actionDim]);
                return tf.randomNormal([this.latentDim, this.actionDim]).mul(baseStd) as tf.Tensor2D;
            });
        }
    }

    /**
     * Пересэмплирование по расписанию (вызывать перед каждым шагом rollout)
     */
    maybeResample() {
        if (this.step % this.noiseUpdateFrequency === 0) {
            this.resample();
        }
        this.step += 1;
    }

    /**
     * Генерация state-dependent шума
     * @param phi - gSDE features из сети, размерность [batchSize, latentDim]
     * @returns шум размерности [batchSize, actionDim]
     */
    noise(phi: tf.Tensor2D): tf.Tensor2D {
        if (!this.Theta) {
            this.resample();
        }

        const raw = phi.matMul(this.Theta!) as tf.Tensor2D; // [B, A]

        if (this.variantA) {
            // Применяем базовое std после умножения
            const baseStd = tf.exp(this.logStdBase).reshape([1, this.actionDim]);
            return raw.mul(baseStd) as tf.Tensor2D;
        }

        return raw;
    }

    resetStep() {
        this.step = 0;
    }

    dispose() {
        this.Theta?.dispose();
    }
}
