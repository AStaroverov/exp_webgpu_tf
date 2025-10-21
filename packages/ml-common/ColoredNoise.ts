import * as tf from '@tensorflow/tfjs';

export class ColoredNoise {
    private state: tf.Tensor;
    private step = 0;

    constructor(actionDim: number, private rho = 0.7, private resetInterval = 30) {
        this.state = tf.randomNormal([actionDim]);
    }

    sample(): tf.Tensor {
        if (++this.step % this.resetInterval === 0) {
            this.state.dispose();
            this.state = tf.randomNormal(this.state.shape)
        }
        // z ~ N(0, I)
        const z = tf.randomNormal(this.state.shape);
        const a = this.rho;
        const b = Math.sqrt(1 - a * a);

        const eps = tf.add(this.state.mul(a), z.mul(b));

        this.state.dispose();
        this.state = eps;

        return eps;
    }

    dispose() {
        this.state.dispose();
    }
}
