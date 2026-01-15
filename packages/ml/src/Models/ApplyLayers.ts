import * as tf from '@tensorflow/tfjs';
import { SymbolicTensor } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { DenseLayerArgs } from '@tensorflow/tfjs-layers/dist/layers/core';
import { MaskLikeLayer } from './Layers/MaskLikeLayer.ts';
import { MultiHeadAttentionLayer } from './Layers/MultiHeadAttentionLayer.ts';
import { RMSNormConfig, RMSNormLayer } from "./Layers/RMSNormLayer.ts";
import { NoisyDenseLayer } from './Layers/NoisyDenseLayer.ts';
import { CloneLayer } from './Layers/CloneLayer.ts';
import { SwinAttentionLayer } from './Layers/SwinAttentionLayer.ts';


export function applyMLP({name, layers: hiddenLayers, preNorm = false}: {
    name: string,
    layers: [ActivationIdentifier, number][],
    preNorm?: boolean
}, layer: tf.SymbolicTensor) {
    if (preNorm) {
        layer = createNormalizationLayer({
            name: `${name}/MLP_preNorm`,
        }).apply(layer) as tf.SymbolicTensor;
    }

    let i = 0;
    for (const [activation, units] of hiddenLayers) {
        layer = createDenseLayer({
            name: `${name}/MLP_dense${i++}`,
            units,
            activation,
            useBias: true
        }).apply(layer) as tf.SymbolicTensor;
    }

    return layer;
}

export function createDenseLayer(options: DenseLayerArgs & Required<Pick<DenseLayerArgs, 'useBias' | 'activation'>> & { noisy?: boolean, sigma?: number }) {
    return options.noisy
        ? new NoisyDenseLayer(options)
        : tf.layers.dense(options);
}

// https://arxiv.org/html/2406.09079v1
export function applyLaNLayer({name, units, preNorm = false}: {
    name: string,
    units: number,
    preNorm?: boolean,
}, layer: tf.SymbolicTensor) {
    if (preNorm) {
        layer = createNormalizationLayer({
            name: `${name}/LaN_preNorm`,
        }).apply(layer) as tf.SymbolicTensor;
    }

    const branch1 = createDenseLayer({
        name: `${name}/LaN_branch1`,
        units,
        useBias: true,
        activation: 'tanh',
    }).apply(layer) as tf.SymbolicTensor;
    const branch2 = createDenseLayer({
        name: `${name}/LaN_branch2`,
        units,
        useBias: true,
        activation: 'tanh',
    }).apply(layer) as tf.SymbolicTensor;

    return tf.layers.multiply({name: `${name}/LaN_output`}).apply([branch1, branch2]) as tf.SymbolicTensor;
}

export function applyNoisyLaNLayer({name, units, sigma, preNorm = false}: {
    name: string,
    units: number,
    sigma?: number,
    preNorm?: boolean,
}, layer: tf.SymbolicTensor) {
    if (preNorm) {
        layer = createNormalizationLayer({
            name: `${name}/NoisyLaN_preNorm`,
        }).apply(layer) as tf.SymbolicTensor;
    }

    const branch1 = new NoisyDenseLayer({
        name: `${name}/NoisyLaN_branch1`,
        units,
        useBias: true,
        activation: 'tanh',
        sigma,
    }).apply(layer) as tf.SymbolicTensor;
    const branch2 = new NoisyDenseLayer({
        name: `${name}/NoisyLaN_branch2`,
        units,
        useBias: true,
        activation: 'tanh',
        sigma,
    }).apply(layer) as tf.SymbolicTensor;

    return tf.layers.multiply({name: `${name}/NoisyLaN_output`}).apply([branch1, branch2]) as tf.SymbolicTensor;
}

export function createNormalizationLayer(options: RMSNormConfig) {
    return new RMSNormLayer(options);
}

export function tokenProj(x: tf.SymbolicTensor, dModel: number, name: string): SymbolicTensor {
   return createDenseLayer({
        name: name + '_tokProj',
        units: dModel,
        useBias: false,
        activation: 'linear'
    }).apply(x) as SymbolicTensor;
}

export function applyCrossAttentionLayer(
    {
        name,
        heads,
        qTok,
        kvTok,
        qMask,
        kvMask,
        preNorm = false,
    }: {
        name: string,
        heads: number,
        qTok: tf.SymbolicTensor,
        qMask?: tf.SymbolicTensor,
        kvTok: tf.SymbolicTensor,
        kvMask?: tf.SymbolicTensor,
        preNorm?: boolean,
    },
) {
    const dModel = qTok.shape[qTok.shape.length - 1]!;
    const isSameToken = qTok === kvTok;
    qTok = preNorm
        ? createNormalizationLayer({ name: name + '_QNorm_' }).apply(qTok) as tf.SymbolicTensor
        : new CloneLayer({ name: name + '_QClone_' }).apply(qTok) as tf.SymbolicTensor;
    kvTok = isSameToken
        ? qTok
        : preNorm
            ? createNormalizationLayer({ name: name + '_KVNorm_' }).apply(kvTok) as tf.SymbolicTensor
            : kvTok;

    // Create mask-like layers if masks are not provided
    qMask ??= new MaskLikeLayer({ name: name + '_qMaskLike' }).apply(qTok) as tf.SymbolicTensor;
    kvMask ??= new MaskLikeLayer({ name: name + '_kvMaskLike' }).apply(kvTok) as tf.SymbolicTensor;

    const attention = new MultiHeadAttentionLayer({
        name: name + '_MultiHeadAttentionLayer',
        keyDim: dModel / heads,
        numHeads: heads,
    }).apply([qTok, qMask, kvTok, kvMask]) as tf.SymbolicTensor;

    return attention;
}

export function applySelfTransformerLayer(
    {
        name,
        heads,
        token,
        mask,
        noisy = false,
        preNorm = false,
    }: {
        name: string,
        heads: number,
        token: tf.SymbolicTensor;
        mask?: tf.SymbolicTensor;
        noisy?: boolean,
        preNorm?: boolean;
    },
) {
    const dModel = token.shape[token.shape.length - 1]!;

    const selfAttn = applyCrossAttentionLayer({
        name,
        heads,
        qTok: token,
        qMask: mask,
        kvTok: token,
        kvMask: mask,
        preNorm
    });

    const attnResidual = tf.layers.add({name: `${name}_residual`})
        .apply([token, selfAttn]) as tf.SymbolicTensor;

    const ffnNorm = createNormalizationLayer({
        name: `${name}_ln2`,
    }).apply(attnResidual) as tf.SymbolicTensor;

    const ffnInner = createDenseLayer({
        name: `${name}_ffn1`,
        units: dModel * 4,
        useBias: false,
        activation: 'relu',
        noisy,
    }).apply(ffnNorm) as tf.SymbolicTensor;

    const ffnOut = createDenseLayer({
        name: `${name}_ffn2`,
        units: dModel,
        useBias: false,
        activation: 'linear',
        noisy,
    }).apply(ffnInner) as tf.SymbolicTensor;

    const finalOut = tf.layers.add({name: `${name}_ffnAdd`})
        .apply([attnResidual, ffnOut]) as tf.SymbolicTensor;

    return finalOut;
}

export function applySelfTransformLayers(name: string, {
    depth,
    heads,
    token,
    mask,
    noisy = false,
    preNorm = false,
}: {
    depth: number,
    heads: number,
    token: tf.SymbolicTensor | ((name: string, i: number) => tf.SymbolicTensor),
    mask?: tf.SymbolicTensor | ((name: string, i: number) => tf.SymbolicTensor),
    noisy?: boolean,
    preNorm?: boolean,
}) {
    let x = typeof token === 'function' ? token(name, 0) : token;
    for (let i = 0; i < depth; i++) {
        const lName = `${name}/depth${i}`;
        x = applySelfTransformerLayer({
            name: lName,
            heads,
            token: x,
            mask: mask ? (typeof mask === 'function' ? mask(lName, i) : mask) : undefined,
            noisy,
            preNorm,
        });
    }

    return x;
}

export function applySwinTransformerLayer(
    {
        name,
        heads,
        token,
        mask,
        window,
        stride,
        noisy = false,
        preNorm = false,
    }: {
        name: string,
        heads: number,
        token: tf.SymbolicTensor,
        mask?: tf.SymbolicTensor,
        window: number,
        stride?: number,
        noisy?: boolean,
        preNorm?: boolean,
    },
) {
    token = preNorm
       ? createNormalizationLayer({ name: name + '_norm_' + token.name}).apply(token) as tf.SymbolicTensor
       : token;
    mask ??= new MaskLikeLayer({ name: token.name + '_maskLike' }).apply(token) as tf.SymbolicTensor;
    
    const dModel = token.shape[token.shape.length - 1]!;
    const clone = new CloneLayer({ name: name + '_clone_' + token.name});
    const selfAttn = new SwinAttentionLayer({
        name: `${name}_SwinAttentionLayer`,
        numHeads: heads,
        keyDim: dModel / heads,
        window,
        stride,
    }).apply([
        clone.apply(token) as tf.SymbolicTensor,
        mask,
        clone.apply(token) as tf.SymbolicTensor,
        mask
    ]) as tf.SymbolicTensor;

    const attnResidual = tf.layers.add({name: `${name}_residual`})
        .apply([token, selfAttn]) as tf.SymbolicTensor;

    const ffnNorm = createNormalizationLayer({
        name: `${name}_ln2`,
    }).apply(attnResidual) as tf.SymbolicTensor;

    const ffnInner = createDenseLayer({
        name: `${name}_ffn1`,
        units: dModel * 4,
        useBias: false,
        activation: 'relu',
        noisy,
    }).apply(ffnNorm) as tf.SymbolicTensor;

    const ffnOut = createDenseLayer({
        name: `${name}_ffn2`,
        units: dModel,
        useBias: false,
        activation: 'linear',
        noisy,
    }).apply(ffnInner) as tf.SymbolicTensor;

    const finalOut = tf.layers.add({name: `${name}_ffnAdd`})
        .apply([attnResidual, ffnOut]) as tf.SymbolicTensor;

    return finalOut;
}

export function applyGlobalAverage1d({ name }: { name: string }, token: tf.SymbolicTensor) {
    return tf.layers.globalAveragePooling1d({ name: name + '_GlobalAvgPool1D' })
        .apply(token) as tf.SymbolicTensor;
}