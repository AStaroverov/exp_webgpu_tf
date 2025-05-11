import * as tf from '@tensorflow/tfjs';

export class FixedPositionalEncodingLayer extends tf.layers.Layer {
    static readonly className = 'FixedPositionalEncodingLayer';

    private encoding!: tf.Tensor3D;

    dispose() {
        this.encoding.dispose();
        return super.dispose();
    }

    build(inputShape: tf.Shape | tf.Shape[]) {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;

        const [, N, dModel] = shape as [number, number, number];

        this.encoding = this.createEncoding(N, dModel); // [1, length, dModel]

        this.built = true;
    }

    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
        const input = Array.isArray(inputs) ? inputs[0] : inputs;
        // broadcasting addition: [B, N, dModel] + [1, N, dModel]
        return tf.add(input, this.encoding);
    }

    private createEncoding(length: number, dModel: number): tf.Tensor3D {
        const position = tf.range(0, length, 1).expandDims(1);  // [length, 1]
        const divTerm = tf.exp(
            tf.mul(
                tf.range(0, dModel, 2).div(dModel),
                tf.scalar(-Math.log(10000.0)),
            ),
        ); // [dModel/2]
        const angleRates = tf.matMul(position.toFloat(), divTerm.expandDims(0)); // [length, dModel/2]
        const sin = tf.sin(angleRates);
        const cos = tf.cos(angleRates);
        const interleaved = tf.concat([sin, cos], 1); // [length, dModel]

        return interleaved.expandDims(0); // [1, length, dModel]
    }
}

tf.serialization.registerClass(FixedPositionalEncodingLayer);
