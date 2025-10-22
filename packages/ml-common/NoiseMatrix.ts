import * as tf from '@tensorflow/tfjs';

export class NoiseMatrix {
    private Theta?: tf.Tensor2D; // [latentDim, actionDim]
    private step = 0;

    constructor(
        private latentDim: number,
        private actionDim: number,
        private noiseUpdateFrequency: number,
    ) {
    }

    resample() {
        this.Theta?.dispose();
        this.Theta = tf.randomNormal([this.latentDim, this.actionDim])
    }

    maybeResample() {
        if (this.step++ % this.noiseUpdateFrequency === 0 || !this.Theta) {
            this.resample();
        }
    }

    noise(logStd: tf.Tensor, phi: tf.Tensor2D): tf.Tensor {
        if (!this.Theta) {
            throw new Error('NoiseMatrix: Theta is not initialized. Call resample() first.');
        }

        return phi.matMul(this.Theta.mul(tf.exp(logStd).reshape([1, this.actionDim]))); // [B, A]
    }

    dispose() {
        this.Theta?.dispose();
    }
}
