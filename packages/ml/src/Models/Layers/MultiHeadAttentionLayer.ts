import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

interface MHAConfig extends LayerArgs {
    numHeads: number;
    keyDim: number;
    // If true, qTok and kvTok are the same tensor and Q/K/V projections
    // are computed via a single fused matmul with weight [D, 3*D].
    selfAttn?: boolean;
}

export class MultiHeadAttentionLayer extends tf.layers.Layer {
    static readonly className = 'MultiHeadAttention';

    private numHeads: number;
    private keyDim: number;
    private selfAttn: boolean;

    // Split path weights
    private wq?: tf.LayerVariable;
    private wk?: tf.LayerVariable;
    private wv?: tf.LayerVariable;

    // Fused self-attn weight: [D, 3*D]
    private wqkv?: tf.LayerVariable;

    private wo!: tf.LayerVariable;

    constructor(config: MHAConfig) {
        super(config);
        this.keyDim = config.keyDim;
        this.numHeads = config.numHeads;
        this.selfAttn = config.selfAttn ?? false;
    }

    build(inputShape: tf.Shape | tf.Shape[]) {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;

        const dModel = shape[shape.length - 1]!;
        const innerDim = this.numHeads * this.keyDim;

        if (dModel !== innerDim) {
            throw new Error(
                `The input feature dimension (dModel=${dModel}) must be equal to numHeads (${this.numHeads}) * keyDim (${this.keyDim}). ` +
                `Currently, numHeads * keyDim = ${innerDim}.`,
            );
        }

        const initializer = tf.initializers.glorotUniform({});
        if (this.selfAttn) {
            this.wqkv = this.addWeight('wqkv', [dModel, innerDim * 3], 'float32', initializer);
        } else {
            this.wq = this.addWeight('wq', [dModel, innerDim], 'float32', initializer);
            this.wk = this.addWeight('wk', [dModel, innerDim], 'float32', initializer);
            this.wv = this.addWeight('wv', [dModel, innerDim], 'float32', initializer);
        }
        this.wo = this.addWeight('wo', [innerDim, dModel], 'float32', initializer);

        super.build(inputShape);
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]) {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;
        return shape;
    }

    call(inputs: tf.Tensor | tf.Tensor[]) {
        return tf.tidy(() => {
            const inputArray = Array.isArray(inputs) ? inputs : [inputs];

            if (inputArray.length !== 4) {
                throw new Error(`MultiHeadAttentionLayer expects exactly 4 inputs: [qTok, qMask, kvTok, kvMask], got ${inputArray.length}`);
            }

            const [qTok, qMask, kvTok, kvMask] = inputArray;
            const [B, N] = qTok.shape;
            const innerDim = this.numHeads * this.keyDim;

            let q: tf.Tensor, k: tf.Tensor, v: tf.Tensor;
            if (this.selfAttn) {
                // Fused projection: one matmul [B*N, D] @ [D, 3D] → split on last axis.
                const fused = write(qTok, this.wqkv!); // [B, N, 3*innerDim]
                const parts = tf.split(fused, 3, -1);
                q = parts[0];
                k = parts[1];
                v = parts[2];
            } else {
                q = write(qTok, this.wq!);
                k = write(kvTok, this.wk!);
                v = write(kvTok, this.wv!);
            }

            const split = (t: tf.Tensor) =>
                t.reshape([B, t.shape[1]!, this.numHeads, this.keyDim])
                    .transpose([0, 2, 1, 3]); // → [B, H, L, d_k]

            const qh = split(q);
            const kh = split(k);
            const vh = split(v);
            const kvMaskReshaped = kvMask.reshape([B, 1, 1, kvMask.shape[1]!]);
            const qMaskReshaped = qMask.reshape([B, qMask.shape[1]!, 1]);

            // Additive mask: masked positions get -inf before softmax.
            // Post-softmax mul is redundant (softmax already outputs ~0 on -inf lanes).
            // Max-subtract for numerical stability with large scores.
            const rawScores = tf.matMul(qh, kh, false, true).div(Math.sqrt(this.keyDim));
            const maskedScores = rawScores.add(kvMaskReshaped.sub(1).mul(1e9));
            const stableScores = maskedScores.sub(maskedScores.max(-1, true));
            const weights = tf.softmax(stableScores);
            const context = tf.matMul(weights, vh);
            const merged = context.transpose([0, 2, 1, 3]).reshape([B, N, innerDim]);
            const output = write(merged, this.wo);

            // Apply qMask to output (mask out padding in query sequence)
            const result = tf.mul(output, qMaskReshaped);

            return result;
        });
    }

    getConfig() {
        const config = super.getConfig();
        return {
            ...config,
            keyDim: this.keyDim,
            numHeads: this.numHeads,
            selfAttn: this.selfAttn,
        };
    }
}

function write(
    input: tf.Tensor,
    weight: tf.LayerVariable,
): tf.Tensor {
    const [B, L, D] = input.shape;
    const flat = input.reshape([B * L, D]);
    const w = weight.read();
    const projected = tf.matMul(flat, w);
    return projected.reshape([B, L, w.shape[1]!]);
}

tf.serialization.registerClass(MultiHeadAttentionLayer);
