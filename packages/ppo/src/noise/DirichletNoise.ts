import * as tf from '@tensorflow/tfjs';

type DirichletNoiseOpts = {
    alpha?: number;           // concentration parameter (default 0.3)
    epsilon?: number;         // minimum value to avoid division issues
    repeatInterval?: [min: number, max: number];  // range for caching: start trying reset at min, force at max
};

/**
 * Dirichlet noise generator, commonly used for exploration in RL (e.g., AlphaZero).
 * 
 * The Dirichlet distribution is parameterized by concentration α:
 * - α < 1: sparse, peaked samples (high exploration)
 * - α = 1: uniform on simplex
 * - α > 1: concentrated toward center (low exploration)
 * 
 * Typical values for RL:
 * - α ≈ 0.03 for large action spaces (e.g., Go with 19x19 = 361 actions)
 * - α ≈ 0.3 for smaller action spaces
 */
export class DirichletNoise {
    private headDims: number[];
    private alpha: number;
    private epsilon: number;
    private repeatMin: number;
    private repeatMax: number;
    private disposed = false;

    // Caching state
    private cachedSamples: tf.Tensor[] | null = null;
    private sampleCount = 0;

    constructor(headDims: number[], opts: DirichletNoiseOpts = {}) {
        const {
            alpha = 0.3,
            epsilon = 1e-10,
            repeatInterval = [1, 1],
        } = opts;

        this.headDims = headDims;
        this.alpha = alpha;
        this.epsilon = epsilon;
        this.repeatMin = Math.max(1, repeatInterval[0]);
        this.repeatMax = Math.max(this.repeatMin, repeatInterval[1]);
    }

    /**
     * Sample Dirichlet noise for each head dimension.
     * Returns array of tensors, each summing to 1.
     * 
     * Uses caching based on repeatInterval [min, max]:
     * - Before min: always returns cached sample
     * - From min to max: probability of reset grows linearly (0 → 1)
     * - At max: forced reset
     */
    sample(): tf.Tensor[] {
        if (this.disposed) throw new Error('DirichletNoise: disposed');

        // Check if we need to generate new samples
        const needNewSample = this.cachedSamples === null || this.shouldResetCache();

        if (needNewSample) {
            this.resetCache();
            this.cachedSamples = this.generateFreshSample();
            this.sampleCount = 0;
        }

        this.sampleCount++;

        // Return clones of cached tensors to prevent external modification
        return this.cachedSamples!.map(t => t.clone());
    }

    /**
     * Check if cache should be reset based on interval range.
     * - Before min: never reset (probability = 0)
     * - At min to max: probability grows linearly from 0 to 1
     * - At max: always reset (probability = 1)
     */
    private shouldResetCache(): boolean {
        if (this.sampleCount < this.repeatMin) {
            return false;
        }
        if (this.sampleCount >= this.repeatMax) {
            return true;
        }
        // Linear interpolation: probability grows from 0 at min to 1 at max
        const range = this.repeatMax - this.repeatMin;
        const progress = (this.sampleCount - this.repeatMin) / range;
        return Math.random() < progress;
    }

    /**
     * Generate fresh Dirichlet samples for all heads
     */
    private generateFreshSample(): tf.Tensor[] {
        return this.headDims.map(dim => {
            return tf.tidy(() => {
                const gamma = this.sampleGamma([dim], this.alpha);
                const sum = gamma.sum().add(this.epsilon);
                return gamma.div(sum);
            });
        });
    }

    /**
     * Clear cached samples and free memory
     */
    private resetCache(): void {
        if (this.cachedSamples) {
            this.cachedSamples.forEach(t => t.dispose());
            this.cachedSamples = null;
        }
    }

    /**
     * Force reset of cached samples (useful for manual control)
     */
    forceReset(): void {
        this.resetCache();
        this.sampleCount = 0;
    }

    /**
     * Sample Gamma(alpha, 1) using Marsaglia and Tsang's method.
     * For alpha < 1, uses the boost: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
     */
    private sampleGamma(shape: number[], alpha: number): tf.Tensor {
        if (alpha < 1) {
            // Boost trick: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
            const gamma1 = this.sampleGammaCore(shape, alpha + 1);
            const u = tf.randomUniform(shape, this.epsilon, 1);
            return gamma1.mul(u.pow(1 / alpha));
        }
        return this.sampleGammaCore(shape, alpha);
    }

    /**
     * Core Marsaglia-Tsang method for Gamma(alpha, 1) where alpha >= 1.
     * Uses acceptance-rejection sampling approximated with fixed iterations.
     */
    private sampleGammaCore(shape: number[], alpha: number): tf.Tensor {
        const d = alpha - 1 / 3;
        const c = 1 / Math.sqrt(9 * d);

        // For tensor operations, we use rejection sampling with oversampling
        // and take the first valid samples. In practice, for small tensors
        // and reasonable alpha values, 8-16 iterations are usually sufficient.
        const oversample = 16;
        const totalShape = [oversample, ...shape];

        // Generate candidate samples
        const z = tf.randomNormal(totalShape);
        const u = tf.randomUniform(totalShape, this.epsilon, 1);

        const v = tf.add(1, tf.mul(c, z));
        const vCubed = tf.pow(v, 3);
        const x = tf.mul(d, vCubed);

        // Acceptance condition: 
        // u < 1 - 0.0331*(z^2)^2 OR log(u) < 0.5*z^2 + d*(1 - v^3 + log(v^3))
        const zSq = tf.square(z);
        const logU = tf.log(u);
        const logVCubed = tf.mul(3, tf.log(tf.maximum(v, this.epsilon)));

        const condition1 = tf.less(u, tf.sub(1, tf.mul(0.0331, tf.square(zSq))));
        const condition2 = tf.less(
            logU,
            tf.add(tf.mul(0.5, zSq), tf.mul(d, tf.sub(tf.add(1, logVCubed), vCubed)))
        );
        const validV = tf.greater(v, 0);
        const accept = tf.logicalAnd(validV, tf.logicalOr(condition1, condition2));

        // For each position in the output, find the first accepted sample
        // We use a simple approach: take weighted average of accepted samples
        // or fallback to the mean approximation
        const acceptFloat = accept.cast('float32');
        const acceptedX = tf.mul(x, acceptFloat);

        // Sum accepted values and count acceptances per position
        const sumX = acceptedX.sum(0);
        const countAccept = tf.maximum(acceptFloat.sum(0), 1); // avoid div by 0

        return tf.div(sumX, countAccept);
    }

    dispose() {
        if (this.disposed) return;
        this.disposed = true;
        this.resetCache();
    }
}

