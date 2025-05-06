import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

export interface MHSAArgs extends LayerArgs {
    keyDim: number;           // d_k  (= dModel / H)
    numHeads: number;          // H
    numKvHeads?: number;        // H (для MQA)
}

export class MultiHeadSelfAttentionLayer extends tf.layers.Layer {
    static className = 'MultiHeadSelfAttentionLayer';
    private readonly keyDim: number;
    private readonly numHeads: number;
    private readonly numKvHeads: number;
    private readonly scale: number;
    private wq!: tf.LayerVariable;
    // private wk!: tf.LayerVariable;
    // private wv!: tf.LayerVariable;
    private wkv!: tf.LayerVariable;

    // private wo!: tf.LayerVariable;

    constructor(config: MHSAArgs) {
        super(config);

        this.scale = Math.sqrt(config.keyDim);
        this.keyDim = config.keyDim;
        this.numHeads = config.numHeads;
        this.numKvHeads = config.numKvHeads ?? Math.min(config.numHeads, 2);

        if (this.numHeads % this.numKvHeads !== 0) {
            throw new Error('numHeads must be divisible by numKvHeads');
        }
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
        this.wq = this.addWeight('wq', [dModel, dModel], 'float32', init);
        // this.wk = this.addWeight('wk', [dModel, dModel], 'float32', init);
        // this.wv = this.addWeight('wv', [dModel, dModel], 'float32', init);
        this.wkv = this.addWeight('wqkv', [dModel, 2 * this.numKvHeads * this.keyDim], 'float32', init);
        // this.wo = this.addWeight('wo', [dModel, dModel], 'float32', init);

        this.built = true;
    }

    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
        const [tokens, mask] = Array.isArray(inputs)
            ? inputs as [tf.Tensor, tf.Tensor?]
            : [inputs as tf.Tensor, undefined];

        const [B, S, dModel] = tokens.shape;
        const G = this.numKvHeads;

        /* ---- flat projection ----- */
        const flat = tokens.reshape([-1, dModel]);            // [B·S, d]

        /* ---------- Q ----------- */
        const Q = tf.matMul(flat, this.wq.read())          // [B·S, d]
            .reshape([B, S, this.numHeads, this.keyDim])
            .transpose([0, 2, 1, 3]);             // [B,H,S,d_k]

        /* ---------- K | V (G голов) ----------- */
        const kv = tf.matMul(flat, this.wkv.read())        // [B·S, 2·G·d_k]
            .reshape([B, S, 2, G, this.keyDim]) // [B,S,2,G,d_k]
            .transpose([2, 0, 3, 1, 4]);        // 2 × [B,G,S,d_k]

        const [K, V] = tf.unstack(kv, 0);                  // каждая: [B,G,S,d_k]

        const repeat = this.numHeads / this.numKvHeads;    // = H/G
        const kh = repeat === 1 ? K : tf.tile(K, [1, repeat, 1, 1]);
        const vh = repeat === 1 ? V : tf.tile(V, [1, repeat, 1, 1]);

        /* ----- scaled dot-product attention ----- */
        // tf.matMul делает broadcast: K/V [B,1,S,d_k] «растягиваются» на H
        // const kh = K.reshape([B, 1, S, this.keyDim]);      // [B,1,S,d_k]
        // const vh = V.reshape([B, 1, S, this.keyDim]);      // [B,1,S,d_k]

        let scores = tf.matMul(Q, kh, false, true)         // [B,H,S,S]
            .div(this.scale);

        // const Q = tf.matMul(flat, this.wq.read()).reshape([B, S, dModel]);
        // const K = tf.matMul(flat, this.wk.read()).reshape([B, S, dModel]);
        // const V = tf.matMul(flat, this.wv.read()).reshape([B, S, dModel]);
        ///* ---- split heads [B,H,S,d_k] ---- */
        // const splitH = (t: tf.Tensor) =>
        //     t.reshape([B, S, this.numHeads, this.keyDim]).transpose([0, 2, 1, 3]);
        //
        // const qh = splitH(Q);
        // const kh = splitH(K);
        // const vh = splitH(V);
        //
        // /* ---- scaled dot-product ---- */
        // let scores = tf.matMul(qh, kh, false, true).div(this.scale); // [B,H,S,S]

        if (mask) {
            const m4 = mask.reshape([B, 1, 1, S]);           // [B,1,1,S]
            scores = scores.add(m4.sub(1).mul(-1e9));
        }

        const ctx = tf.matMul(tf.softmax(scores), vh);      // [B,H,S,d_k]

        /* ----- merge heads + вернуть W_O проекцию ----- */
        const merged = ctx
            .transpose([0, 2, 1, 3])         // [B,S,H,d_k]
            .reshape([B, S, dModel]);       // [B,S,d]

        return merged;
        // /* ---------- final linear proj W_O  (с флэтом!) ------ */
        // const flatOut = merged.reshape([-1, dModel]);          // [B·S, d]
        // const proj = tf.matMul(flatOut, this.wo.read());       // [B·S, d]
        // return proj.reshape([B, S, dModel]);                   // [B,S,dModel]
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

// import * as tf from '@tensorflow/tfjs';
// import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
//
// export interface MHSAArgs extends LayerArgs {
//     numHeads: number;   // H  (будет пере-проверено, чтобы d_k ≥ 16)
//     keyDim: number;    // d_k (= dModel / H)
//     useBias?: boolean;
// }
//
// export class MultiHeadSelfAttentionLayer extends tf.layers.Layer {
//     static readonly className = 'MultiHeadSelfAttentionLayer';
//
//     private numHeads!: number;                  // H (выровненный)
//     private keyDim!: number;                   // d_k
//     private scale!: number;                   // √d_k
//
//     private denseQ!: tf.layers.Layer;
//     private denseK!: tf.layers.Layer;
//     private denseV!: tf.layers.Layer;
//     private denseO!: tf.layers.Layer;
//
//     constructor(cfg: MHSAArgs) {
//         super(cfg);
//         this.keyDim = cfg.keyDim;
//         this.numHeads = cfg.numHeads;
//         this.scale = Math.sqrt(this.keyDim);
//     }
//
//     computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape {
//         const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
//             ? inputShape
//             : inputShape[0]) as tf.Shape;
//         return shape;
//     }
//
//     build(inputShape: tf.Shape | tf.Shape[]): void {
//         const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
//             ? inputShape
//             : inputShape[0]) as tf.Shape;
//         const dModel = shape[shape.length - 1] as number;
//
//         const makeDense = (name: string) =>
//             tf.layers.dense({
//                 name: name,
//                 units: dModel,
//                 useBias: false,
//                 kernelInitializer: 'glorotUniform',
//             });
//
//         this.denseQ = makeDense(this.name + '_denseQ');
//         this.denseK = makeDense(this.name + '_denseK');
//         this.denseV = makeDense(this.name + '_denseV');
//         this.denseO = makeDense(this.name + '_denseO');
//
//         this.built = true;
//     }
//
//     call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
//         const [tokens, mask] = Array.isArray(inputs)
//             ? inputs as [tf.Tensor, tf.Tensor?]
//             : [inputs as tf.Tensor, undefined];
//
//         const [B, S, dModel] = tokens.shape;
//
//         /* ---- Q / K / V : [B,S,d]  (через fused-Dense) ------------------- */
//         const Q = this.denseQ.apply(tokens) as tf.Tensor;
//         const K = this.denseK.apply(tokens) as tf.Tensor;
//         const V = this.denseV.apply(tokens) as tf.Tensor;
//
//         /* ---- split heads → [B,H,S,d_k] ---------------------------------- */
//         const split = (t: tf.Tensor) =>
//             t.reshape([B, S, this.numHeads, this.keyDim])
//                 .transpose([0, 2, 1, 3]);                        // [B,H,S,d_k]
//
//         const qh = split(Q);
//         const kh = split(K);
//         const vh = split(V);
//
//         /* ---- scaled dot-product ---------------------------------------- */
//         let scores = tf.matMul(qh, kh, false, true)      // [B,H,S,S]
//             .div(this.scale);
//
//         if (mask) {
//             const m4 = tf.expandDims(tf.expandDims(mask, 1), 1);  // [B,1,1,S]
//             scores = scores.add(m4.sub(1).mul(tf.scalar(-1e9)));
//         }
//
//         const weights = tf.softmax(scores);              // [B,H,S,S]
//         const ctx = tf.matMul(weights, vh);          // [B,H,S,d_k]
//
//         /* ---- merge heads back → [B,S,d] -------------------------------- */
//         const merged = ctx.transpose([0, 2, 1, 3])          // [B,S,H,d_k]
//             .reshape([B, S, dModel]);
//
//         /* ---- O-проекция (Dense) ---------------------------------------- */
//         return this.denseO.apply(merged) as tf.Tensor;   // [B,S,d]
//     }
//
//     getConfig(): tf.serialization.ConfigDict {
//         return {
//             ...super.getConfig(),
//             keyDim: this.keyDim,
//             numHeads: this.numHeads,
//         };
//     }
// }
//
// tf.serialization.registerClass(MultiHeadSelfAttentionLayer);
