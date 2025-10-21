import * as tf from '@tensorflow/tfjs';

export class NoiseMatrix {
    private Theta?: tf.Tensor2D; // [latentDim, actionDim]
    private step = 0;

    constructor(
        private latentDim: number,
        private actionDim: number,
        private noiseUpdateFrequency: number,
        public logStdBase: tf.Variable,
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

    maybeResample() {
        if (this.step++ % this.noiseUpdateFrequency === 0 || !this.Theta) {
            this.resample();
        }
    }

    noise(phi: tf.Tensor2D): tf.Tensor2D {
        if (!this.Theta) {
            throw new Error('NoiseMatrix: Theta is not initialized. Call resample() first.');
        }

        const raw = phi.matMul(this.Theta!) as tf.Tensor2D; // [B, A]

        if (this.variantA) {
            // Применяем базовое std после умножения
            const baseStd = tf.exp(this.logStdBase).reshape([1, this.actionDim]);
            return raw.mul(baseStd) as tf.Tensor2D;
        }

        return raw;
    }

    dispose() {
        this.Theta?.dispose();
    }
}
