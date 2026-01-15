import * as tf from '@tensorflow/tfjs';
import { applyCrossAttentionLayer, createNormalizationLayer, createDenseLayer } from "../ApplyLayers";

export function applyCrossTransformerLayer(
    {
        name,
        heads,
        qTok,
        qMask,
        kvTok,
        kvMask,
        preNorm = false,
    }: {
        name: string,
        heads: number,
        qTok: tf.SymbolicTensor,
        kvTok: tf.SymbolicTensor,
        qMask?: tf.SymbolicTensor,
        kvMask?: tf.SymbolicTensor,
        preNorm?: boolean,
    },
) {
    const dModel = qTok.shape[qTok.shape.length - 1]!;

    const crossAttn = applyCrossAttentionLayer({name, heads, qTok, qMask, kvTok, kvMask, preNorm});

    const attnResidual = tf.layers.add({name: `${name}_residual`})
        .apply([qTok, crossAttn]) as tf.SymbolicTensor;

    const ffnNorm = createNormalizationLayer({
        name: `${name}_ffnLN`,
    }).apply(attnResidual) as tf.SymbolicTensor;

    const ffnInner = createDenseLayer({
        name: `${name}_ffn1`,
        units: dModel * 4,
        useBias: false,
        activation: 'relu',
    }).apply(ffnNorm) as tf.SymbolicTensor;

    const ffnOut = createDenseLayer({
        name: `${name}_ffn2`,
        units: dModel,
        useBias: false,
        activation: 'linear'
    }).apply(ffnInner) as tf.SymbolicTensor;

    const finalOut = tf.layers.add({name: `${name}_ffnAdd`})
        .apply([attnResidual, ffnOut]) as tf.SymbolicTensor;

    return finalOut;
}
