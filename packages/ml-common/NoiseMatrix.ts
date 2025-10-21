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

    resample() {
        this.Theta?.dispose();
        this.Theta = tf.tidy(() => {
            const baseStd = tf.exp(this.logStdBase).reshape([1, this.actionDim]);
            return tf.randomNormal([this.latentDim, this.actionDim]).mul(baseStd) as tf.Tensor2D;
        });
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

        return phi.matMul(this.Theta!) as tf.Tensor2D; // [B, A]
    }

    dispose() {
        this.Theta?.dispose();
    }
}
