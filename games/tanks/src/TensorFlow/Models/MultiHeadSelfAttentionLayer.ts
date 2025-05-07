import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

export interface MHSAArgs extends LayerArgs {
    keyDim: number;
    numHeads: number;
}

export class MultiHeadSelfAttentionLayer extends tf.layers.Layer {
    static className = 'MultiHeadSelfAttentionLayer';

    private readonly numHeads: number;
    private readonly keyDim: number;
    private readonly scale: number;
    private wqkv!: tf.LayerVariable;
    private wo!: tf.LayerVariable;

    constructor(config: MHSAArgs) {
        super(config);
        this.keyDim = config.keyDim;
        this.numHeads = config.numHeads;
        this.scale = Math.sqrt(this.keyDim);
    }

    /** Shape doesn’t change: [B,S,dModel] → [B,S,dModel] */
    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;
        return shape;
    }

    /** Create variables W_Q, W_K, W_V, W_O  (all dModel × dModel) */
    build(inputShape: tf.Shape | tf.Shape[]): void {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;
        const dModel = shape[shape.length - 1] as number;

        const init = tf.initializers.glorotUniform({});
        this.wqkv = this.addWeight('wqkv', [dModel, 3 * dModel], 'float32', init);
        this.wo = this.addWeight('wo', [dModel, dModel], 'float32', init);

        this.built = true;
    }

    dispose() {
        this.wqkv.dispose();
        this.wo.dispose();
        return super.dispose();
    }

    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
        return tf.tidy(() => {
            const [tokens, mask] = Array.isArray(inputs)
                ? inputs as [tf.Tensor, tf.Tensor?]
                : [inputs as tf.Tensor, undefined];

            const [B, S, dModel] = tokens.shape;

            /* ---- flat projection ----- */
            const flat = tokens.reshape([-1, dModel]);            // [B·S, d]

            const qkv = tf.matMul(flat, this.wqkv.read())              // [B·S, 3d]
                .reshape([B, S, 3, this.numHeads, this.keyDim])
                .transpose([2, 0, 3, 1, 4]);                // 3 × B × H × S × d_k

            /* ---- split heads [B,H,S,d_k] ---- */
            const [qh, kh, vh] = tf.unstack(qkv, 0);

            /* ---- scaled dot-product ---- */
            let scores = tf.matMul(qh, kh, false, true).div(this.scale); // [B,H,S,S]

            if (mask) {
                const m4 = mask.reshape([B, 1, 1, S]);
                scores = scores.add(m4.sub(1).mul(1e9));
            }

            const weights = tf.softmax(scores);          // [B,H,S,S]
            const ctx = tf.matMul(weights, vh);      // [B,H,S,d_k]

            /* ---------- merge heads back: [B,S,dModel] ---------- */
            const merged = ctx
                .transpose([0, 2, 1, 3])        // [B,S,H,d_k]
                .reshape([B, S, dModel]);       // [B,S,d]

            /* ---------- final linear proj W_O ------ */
            const flatOut = merged.reshape([-1, dModel]);          // [B·S, d]
            const proj = tf.matMul(flatOut, this.wo.read());       // [B·S, d]
            return proj.reshape([B, S, dModel]);                   // [B,S,dModel]
        });
    }

    /** Save & load support */
    getConfig(): tf.serialization.ConfigDict {
        const base = super.getConfig();
        return {
            ...base,
            keyDim: this.keyDim,
            numHeads: this.numHeads,
        };
    }
}

tf.serialization.registerClass(MultiHeadSelfAttentionLayer);
