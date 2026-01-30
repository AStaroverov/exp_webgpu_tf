import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

interface MHAConfig extends LayerArgs {
    numHeads: number;
    keyDim: number;
}

export class MultiHeadAttentionLayer extends tf.layers.Layer {
    static readonly className = 'MultiHeadAttention';

    private numHeads: number;
    private keyDim: number;
    private wq!: tf.LayerVariable;
    private wk!: tf.LayerVariable;
    private wv!: tf.LayerVariable;
    private wo!: tf.LayerVariable;

    constructor(config: MHAConfig) {
        super(config);
        this.keyDim = config.keyDim;
        this.numHeads = config.numHeads;
    }

    build(inputShape: tf.Shape | tf.Shape[]) {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;

        const dModel = shape[shape.length - 1]!;

        if (dModel !== this.numHeads * this.keyDim) {
            throw new Error(
                `The input feature dimension (dModel=${dModel}) must be equal to numHeads (${this.numHeads}) * keyDim (${this.keyDim}). ` +
                `Currently, numHeads * keyDim = ${this.numHeads * this.keyDim}.`,
            );
        }

        const initializer = tf.initializers.glorotUniform({});
        this.wq = this.addWeight('wq', [dModel, this.numHeads * this.keyDim], 'float32', initializer);
        this.wk = this.addWeight('wk', [dModel, this.numHeads * this.keyDim], 'float32', initializer);
        this.wv = this.addWeight('wv', [dModel, this.numHeads * this.keyDim], 'float32', initializer);
        this.wo = this.addWeight('wo', [this.numHeads * this.keyDim, dModel], 'float32', initializer);

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
            
            const [qTok, qMask, kvTok,  kvMask] = inputArray;
            const [B, N, dModel] = qTok.shape;

            const q = write(qTok, this.wq);   // [b, qLen, heads*keyDim]
            const k = write(kvTok, this.wk);   // [b, kLen, heads*keyDim]
            const v = write(kvTok, this.wv);   // [b, vLen, heads*keyDim]

            const split = (t: tf.Tensor) =>
                t.reshape([B, t.shape[1]!, this.numHeads, this.keyDim])
                    .transpose([0, 2, 1, 3]); // → [B, H, Q or K, d_k]

            const qh = split(q);
            const kh = split(k);
            const vh = split(v);
            const kvMaskReshaped = kvMask.reshape([B, 1, 1, kvMask.shape[1]!]);
            const qMaskReshaped = qMask.reshape([B, qMask.shape[1]!, 1]);

            const scores = tf.matMul(qh, kh, false, true)
                .div(Math.sqrt(this.keyDim))
                .add(kvMaskReshaped.sub(1).mul(1e5));
            const weights = tf.softmax(scores).mul(kvMaskReshaped);
            const context = tf.matMul(weights, vh);
            const merged = context.transpose([0, 2, 1, 3]).reshape([B, N, dModel]);
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

