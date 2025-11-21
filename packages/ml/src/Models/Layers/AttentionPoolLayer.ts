import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { MultiHeadAttentionLayer } from './MultiHeadAttentionLayer';

interface AttentionPoolConfig extends LayerArgs {
    numHeads: number;
    keyDim: number;
}

export class AttentionPoolLayer extends tf.layers.Layer {
    static className = 'AttentionPool';

    private queryToken!: tf.LayerVariable;
    private attention!: MultiHeadAttentionLayer;
    private numHeads: number;
    private keyDim: number;

    constructor(config: AttentionPoolConfig) {
        super(config);
        this.numHeads = config.numHeads;
        this.keyDim = config.keyDim;
    }

    build(inputShape: tf.Shape | tf.Shape[]) {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;
        const D = shape[2]!;

        if (D !== this.numHeads * this.keyDim) {
            throw new Error(
                `Input feature dimension (${D}) must equal numHeads (${this.numHeads}) * keyDim (${this.keyDim}) = ${this.numHeads * this.keyDim}`
            );
        }

        this.queryToken = this.addWeight(
            'query',
            [1, 1, D],
            'float32',
            tf.initializers.glorotUniform({})
        );

        this.attention = new MultiHeadAttentionLayer({
            numHeads: this.numHeads,
            keyDim: this.keyDim,
        });
        this.attention.build([1, 1, D]);

        this.built = true;
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]) {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;
        return [shape[0], 1, shape[2]];
    }

    call(tokens: tf.Tensor[]) {
        const [batch] = tokens[0].shape;

        // Expand query token to batch size
        const queries = this.queryToken.read().tile([batch, 1, 1]);

        // Use MultiHeadAttention for cross-attention
        const attended = this.attention.apply([queries, ...tokens]) as tf.Tensor;

        return attended;
    }

    getConfig() {
        const config = super.getConfig();
        return {
            ...config,
            numHeads: this.numHeads,
            keyDim: this.keyDim,
        };
    }
}

tf.serialization.registerClass(AttentionPoolLayer);