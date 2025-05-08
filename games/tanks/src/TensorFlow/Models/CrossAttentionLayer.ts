import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

export interface CrossAttentionArgs extends LayerArgs {
    numHeads: number;       // количество голов
    keyDim: number;         // размерность одной головы
    useBias?: boolean;      // по умолчанию true
}

export class CrossAttentionLayer extends tf.layers.Layer {
    static readonly className = 'CrossAttentionLayer';
    private numHeads!: number;
    private keyDim!: number;
    private scale!: number;
    private useBias: boolean;
    private denseQ!: tf.layers.Layer;
    private denseK!: tf.layers.Layer;
    private denseV!: tf.layers.Layer;
    private denseO!: tf.layers.Layer;

    constructor(cfg: CrossAttentionArgs) {
        super(cfg);
        this.numHeads = cfg.numHeads;
        this.keyDim = cfg.keyDim;
        this.useBias = cfg.useBias ?? true;
        this.scale = Math.sqrt(this.keyDim);
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;
        return shape;
    }

    build(inputShape: tf.Shape | tf.Shape[]): void {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;

        const units = shape[shape.length - 1]!;

        if (units !== this.numHeads * this.keyDim) {
            throw new Error(
                `The input feature dimension (dModel=${ units }) must be equal to numHeads (${ this.numHeads }) * keyDim (${ this.keyDim }). ` +
                `Currently, numHeads * keyDim = ${ this.numHeads * this.keyDim }.`,
            );
        }

        const createDense = (name: string, useBias: boolean) =>
            tf.layers.dense({ name, units, useBias });

        this.denseQ = createDense(this.name + '_Q', this.useBias);
        this.denseK = createDense(this.name + '_K', this.useBias);
        this.denseV = createDense(this.name + '_V', this.useBias);
        this.denseO = createDense(this.name + '_O', true);

        this.built = true;
    }

    dispose() {
        this.denseQ.dispose();
        this.denseK.dispose();
        this.denseV.dispose();
        this.denseO.dispose();
        return super.dispose();
    }

    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
        return tf.tidy(() => {
            const [qInput, kvInput, mask] = inputs as [tf.Tensor, tf.Tensor, tf.Tensor?];
            const [B, Q, dModel] = qInput.shape;

            const Q_proj = this.denseQ.apply(qInput) as tf.Tensor;
            const K_proj = this.denseK.apply(kvInput) as tf.Tensor;
            const V_proj = this.denseV.apply(kvInput) as tf.Tensor;

            const split = (t: tf.Tensor) =>
                t.reshape([B, t.shape[1]!, this.numHeads, this.keyDim])
                    .transpose([0, 2, 1, 3]); // → [B, H, Q or K, d_k]

            const qh = split(Q_proj);  // [B,H,Q,d_k]
            const kh = split(K_proj);  // [B,H,K,d_k]
            const vh = split(V_proj);  // [B,H,K,d_k]

            let scores = tf.matMul(qh, kh, false, true).div(this.scale); // [B,H,Q,K]

            if (mask) {
                const shape = mask.shape;
                const expandedMask = mask.reshape([shape[0], 1, 1, shape[1]!]);
                scores = scores.add(expandedMask.sub(1).mul(1e9));
            }

            const weights = tf.softmax(scores);   // [B,H,Q,K]
            const ctx = tf.matMul(weights, vh);   // [B,H,Q,d_k]

            const merged = ctx.transpose([0, 2, 1, 3]).reshape([B, Q, dModel]); // [B,Q,d]

            return this.denseO.apply(merged) as tf.Tensor; // [B,Q,d]
        });
    }

    getConfig(): tf.serialization.ConfigDict {
        return {
            ...super.getConfig(),
            numHeads: this.numHeads,
            keyDim: this.keyDim,
            useBias: this.useBias,
        };
    }
}

tf.serialization.registerClass(CrossAttentionLayer);
