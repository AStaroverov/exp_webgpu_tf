import * as tf from '@tensorflow/tfjs';
import type { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

/**
 * 2D sinusoidal positional encoding for a flattened ROWS×COLS grid of tokens.
 *
 * The 1D `FixedPositionalEncodingLayer` only encodes the linear token index, so
 * on a board it never tells the model that two cells share a row/column or are
 * vertically adjacent. Here we encode `row` and `col` independently — each over
 * half of dModel — and concatenate, so the encoding reflects true 2D geometry.
 *
 * Input/output: [B, ROWS*COLS, dModel]; the encoding [1, ROWS*COLS, dModel] is
 * added with broadcasting. dModel must be divisible by 4 (split in two halves,
 * each needing an even count for the sin/cos pair).
 */
export class Grid2DPositionalEncodingLayer extends tf.layers.Layer {
    static readonly className = 'Grid2DPositionalEncodingLayer';

    private rows: number;
    private cols: number;
    private encoding!: tf.Tensor3D;

    constructor(config: LayerArgs & { rows: number, cols: number }) {
        super(config);
        this.rows = config.rows;
        this.cols = config.cols;
    }

    dispose() {
        this.encoding?.dispose();
        return super.dispose();
    }

    build(inputShape: tf.Shape | tf.Shape[]) {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;

        const [, N, dModel] = shape as [number, number, number];

        if (N !== this.rows * this.cols) {
            throw new Error(
                `Grid2DPositionalEncodingLayer: token count ${N} != rows*cols ` +
                `(${this.rows}*${this.cols}=${this.rows * this.cols}).`,
            );
        }
        if (dModel % 4 !== 0) {
            throw new Error(
                `Grid2DPositionalEncodingLayer: dModel (${dModel}) must be divisible by 4.`,
            );
        }

        this.encoding = this.createEncoding(dModel); // [1, N, dModel]
        this.built = true;
    }

    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
        const input = Array.isArray(inputs) ? inputs[0] : inputs;
        // broadcasting addition: [B, N, dModel] + [1, N, dModel]
        return tf.add(input, this.encoding);
    }

    /** Sinusoidal encoding of `positions` [N] over `dim` (even) → [N, dim]. */
    private sinusoidal(positions: tf.Tensor1D, dim: number): tf.Tensor {
        const pos = positions.toFloat().expandDims(1); // [N, 1]
        const divTerm = tf.exp(
            tf.mul(
                tf.range(0, dim, 2).div(dim),
                tf.scalar(-Math.log(10000.0)),
            ),
        ); // [dim/2]
        const angleRates = tf.matMul(pos, divTerm.expandDims(0)); // [N, dim/2]
        return tf.concat([tf.sin(angleRates), tf.cos(angleRates)], 1); // [N, dim]
    }

    private createEncoding(dModel: number): tf.Tensor3D {
        const half = dModel / 2;
        // row/col index of every cell, cell-major (idx = row*cols + col).
        const rowIdx = tf.range(0, this.rows).expandDims(1).tile([1, this.cols]).reshape([-1]) as tf.Tensor1D;
        const colIdx = tf.range(0, this.cols).expandDims(0).tile([this.rows, 1]).reshape([-1]) as tf.Tensor1D;

        const rowEnc = this.sinusoidal(rowIdx, half); // [N, half]
        const colEnc = this.sinusoidal(colIdx, half); // [N, half]
        const interleaved = tf.concat([rowEnc, colEnc], 1); // [N, dModel]

        return interleaved.expandDims(0); // [1, N, dModel]
    }

    getConfig() {
        const config = super.getConfig();
        return {
            ...config,
            rows: this.rows,
            cols: this.cols,
        };
    }
}

tf.serialization.registerClass(Grid2DPositionalEncodingLayer);
