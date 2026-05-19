import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

interface SwinAttentionConfig extends LayerArgs {
    numHeads: number;
    keyDim: number;
    window: number;  // Size of attention window (e.g., 7)
    stride?: number;     // Stride for sliding windows (default: windowSize for non-overlapping)
}

export class SwinAttentionLayer extends tf.layers.Layer {
    static readonly className = 'SwinAttention';

    private numHeads: number;
    private keyDim: number;
    private window: number;
    private stride: number;
    private wq!: tf.LayerVariable;
    private wk!: tf.LayerVariable;
    private wv!: tf.LayerVariable;
    private wo!: tf.LayerVariable;

    constructor(config: SwinAttentionConfig) {
        super(config);
        this.keyDim = config.keyDim;
        this.numHeads = config.numHeads;
        this.window = config.window;
        this.stride = config.stride ?? config.window; // default: non-overlapping
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
                throw new Error(`SwinAttentionLayer expects exactly 4 inputs: [qTok, qMask, kvTok, kvMask], got ${inputArray.length}`);
            }
            
            const [qTok, qMask, kvTok, kvMask] = inputArray;
            const [B, N, dModel] = qTok.shape;

            // Calculate number of windows with stride
            const numWindows = Math.floor((N - this.window) / this.stride) + 1;
            
            // Extract sliding windows
            const qWindows: tf.Tensor[] = [];
            const kvWindows: tf.Tensor[] = [];
            const qMaskWindows: tf.Tensor[] = [];
            const kvMaskWindows: tf.Tensor[] = [];
            
            for (let i = 0; i < numWindows; i++) {
                const start = i * this.stride;
                const end = Math.min(start + this.window, N);
                const actualWindowSize = end - start;
                
                // Slice windows
                let qWin = qTok.slice([0, start, 0], [B, actualWindowSize, dModel]);
                let kvWin = kvTok.slice([0, start, 0], [B, actualWindowSize, dModel]);
                let qMaskWin = qMask.slice([0, start], [B, actualWindowSize]);
                let kvMaskWin = kvMask.slice([0, start], [B, actualWindowSize]);
                
                // Pad if window is smaller than windowSize
                if (actualWindowSize < this.window) {
                    const padSize = this.window - actualWindowSize;
                    qWin = tf.pad(qWin, [[0, 0], [0, padSize], [0, 0]]);
                    kvWin = tf.pad(kvWin, [[0, 0], [0, padSize], [0, 0]]);
                    qMaskWin = tf.pad(qMaskWin, [[0, 0], [0, padSize]]);
                    kvMaskWin = tf.pad(kvMaskWin, [[0, 0], [0, padSize]]);
                }
                
                qWindows.push(qWin);
                kvWindows.push(kvWin);
                qMaskWindows.push(qMaskWin);
                kvMaskWindows.push(kvMaskWin);
            }
            
            // Stack windows: [B, numWindows, windowSize, dModel]
            const qStacked = tf.stack(qWindows, 1);
            const kvStacked = tf.stack(kvWindows, 1);
            const qMaskStacked = tf.stack(qMaskWindows, 1);
            const kvMaskStacked = tf.stack(kvMaskWindows, 1);
            
            // Flatten batch and windows: [B*numWindows, windowSize, dModel]
            const BW = B * numWindows;
            const qFlat = qStacked.reshape([BW, this.window, dModel]);
            const kvFlat = kvStacked.reshape([BW, this.window, dModel]);
            const qMaskFlat = qMaskStacked.reshape([BW, this.window]);
            const kvMaskFlat = kvMaskStacked.reshape([BW, this.window]);

            // Apply Q, K, V projections
            const q = write(qFlat, this.wq);
            const k = write(kvFlat, this.wk);
            const v = write(kvFlat, this.wv);

            const split = (t: tf.Tensor) =>
                t.reshape([BW, this.window, this.numHeads, this.keyDim])
                    .transpose([0, 2, 1, 3]); // → [BW, H, windowSize, d_k]

            const qh = split(q);
            const kh = split(k);
            const vh = split(v);

            // Compute attention scores
            let scores = tf.matMul(qh, kh, false, true)
                .div(Math.sqrt(this.keyDim)); // [BW, H, windowSize, windowSize]

            // Apply masks
            const kvMaskReshaped = kvMaskFlat.reshape([BW, 1, 1, this.window]);
            const qMaskReshaped = qMaskFlat.reshape([BW, this.window, 1]);
            
            scores = scores.add(kvMaskReshaped.sub(1).mul(1e9));
            
            const weights = tf.softmax(scores).mul(kvMaskReshaped);
            const context = tf.matMul(weights, vh); // [BW, H, windowSize, d_k]
            
            const merged = context.transpose([0, 2, 1, 3])
                .reshape([BW, this.window, dModel]);
            
            let windowOutputs = write(merged, this.wo);
            
            // Apply qMask to output
            windowOutputs = tf.mul(windowOutputs, qMaskReshaped);
            
            // Reshape back: [B, numWindows, windowSize, dModel]
            windowOutputs = windowOutputs.reshape([B, numWindows, this.window, dModel]);
            
            // Aggregate overlapping windows
            // Create output tensor and count tensor for averaging
            let output = tf.zeros([B, N, dModel]);
            let counts = tf.zeros([B, N, 1]);
            
            for (let i = 0; i < numWindows; i++) {
                const start = i * this.stride;
                const end = Math.min(start + this.window, N);
                const actualWindowSize = end - start;
                
                // Extract window output
                const winOut = windowOutputs.slice([0, i, 0, 0], [B, 1, actualWindowSize, dModel])
                    .reshape([B, actualWindowSize, dModel]);
                
                // Add to output at correct position
                const beforePad = tf.zeros([B, start, dModel]);
                const afterPad = tf.zeros([B, N - end, dModel]);
                const paddedWinOut = tf.concat([beforePad, winOut, afterPad], 1);
                output = output.add(paddedWinOut);
                
                // Update counts
                const countPad = tf.ones([B, actualWindowSize, 1]);
                const beforeCountPad = tf.zeros([B, start, 1]);
                const afterCountPad = tf.zeros([B, N - end, 1]);
                const paddedCount = tf.concat([beforeCountPad, countPad, afterCountPad], 1);
                counts = counts.add(paddedCount);
            }
            
            // Average overlapping positions
            output = output.div(counts.maximum(1)); // avoid division by zero
            
            // Apply original qMask to final output
            const finalMask = qMask.expandDims(2);
            output = output.mul(finalMask);

            return output;
        });
    }

    getConfig() {
        const config = super.getConfig();
        return {
            ...config,
            keyDim: this.keyDim,
            numHeads: this.numHeads,
            window: this.window,
            stride: this.stride,
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

tf.serialization.registerClass(SwinAttentionLayer);

