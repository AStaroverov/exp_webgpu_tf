import * as tf from '@tensorflow/tfjs';
import { AttentionMaskLayer } from './AttentionMaskLayer.ts';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';

export function applyAttentionLayer(name: string, qInput: tf.SymbolicTensor, kvInput: tf.SymbolicTensor, kvMaskInput: tf.SymbolicTensor, dModel = 32): tf.SymbolicTensor {
    const denseQ = tf.layers.dense({ name: name + '_denseQ', units: dModel, useBias: false });
    const denseK = tf.layers.dense({ name: name + '_denseK', units: dModel, useBias: false });
    const denseV = tf.layers.dense({ name: name + '_denseV', units: dModel, useBias: false });

    const Q = denseQ.apply(qInput) as tf.SymbolicTensor;  // [batch, dim model]
    const K = denseK.apply(kvInput) as tf.SymbolicTensor;  // [batch, slots, dim model]
    const V = denseV.apply(kvInput) as tf.SymbolicTensor;

    const Q_reshaped = tf.layers.reshape({ targetShape: [1, dModel] }).apply(Q) as tf.SymbolicTensor; // [batch, 1, d_model]

    const scores = tf.layers.dot({ axes: -1 }).apply([Q_reshaped, K]) as tf.SymbolicTensor;

    const maskedScores = new AttentionMaskLayer().apply([scores, kvMaskInput]) as tf.SymbolicTensor;

    const attnWeights = tf.layers.activation({ activation: 'softmax' }).apply(maskedScores) as tf.SymbolicTensor;

    const context = tf.layers.dot({ axes: [2, 1] }).apply([attnWeights, V]) as tf.SymbolicTensor;

    return tf.layers.flatten().apply(context) as tf.SymbolicTensor; // [batch, d_model]
}

export function applyDenseLayers(layer: tf.SymbolicTensor, hiddenLayers: [ActivationIdentifier, number][]) {
    let x = layer;
    let i = 0;
    for (const [activation, units] of hiddenLayers) {
        x = tf.layers.dense({
            name: `${ layer.name }/dense${ i++ }`,
            units,
            activation,
            kernelInitializer: 'glorotUniform',
        }).apply(x) as tf.SymbolicTensor;
    }

    return x;
}