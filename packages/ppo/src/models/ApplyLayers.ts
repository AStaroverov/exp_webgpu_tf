import * as tf from '@tensorflow/tfjs';
import { SymbolicTensor } from '@tensorflow/tfjs-layers/dist/engine/topology';
import type { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import type { DenseLayerArgs } from '@tensorflow/tfjs-layers/dist/layers/core';
import { MaskLikeLayer } from './Layers/MaskLikeLayer.ts';
import { MultiHeadAttentionLayer } from './Layers/MultiHeadAttentionLayer.ts';
import { RMSNormLayer } from "./Layers/RMSNormLayer.ts";
import type { RMSNormConfig } from "./Layers/RMSNormLayer.ts";
import { NoisyDenseLayer } from './Layers/NoisyDenseLayer.ts';

export function tokenProj(x: tf.SymbolicTensor, dModel: number, name: string): SymbolicTensor {
   return createDenseLayer({
        name: name + '_tokProj',
        units: dModel,
        useBias: false,
        activation: 'linear'
    }).apply(x) as SymbolicTensor;
}

export function createNormalizationLayer(options: RMSNormConfig) {
    return new RMSNormLayer(options);
}

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
    if (preNorm) {
        qTok = createNormalizationLayer({ name: name + '_QNorm_' }).apply(qTok) as tf.SymbolicTensor;
        kvTok = isSameToken
            ? qTok
            : createNormalizationLayer({ name: name + '_KVNorm_' }).apply(kvTok) as tf.SymbolicTensor;
    }

    // Create mask-like layers if masks are not provided
    qMask ??= new MaskLikeLayer({ name: name + '_qMaskLike' }).apply(qTok) as tf.SymbolicTensor;
    kvMask ??= new MaskLikeLayer({ name: name + '_kvMaskLike' }).apply(kvTok) as tf.SymbolicTensor;

    // Self-attn: never duplicate a tensor in one input array — it trips tfjs's
    // disposal refcount (see MultiHeadAttentionLayer.call). Same mask → 2
    // inputs; asymmetric masks (all queries, content-only keys) → 3.
    const attention = new MultiHeadAttentionLayer({
        name: name + '_MultiHeadAttentionLayer',
        keyDim: dModel / heads,
        numHeads: heads,
        selfAttn: isSameToken,
    }).apply(
        !isSameToken ? [qTok, qMask, kvTok, kvMask]
            : qMask === kvMask ? [qTok, qMask]
            : [qTok, qMask, kvMask],
    ) as tf.SymbolicTensor;

    return attention;
}

/**
 * Apply a pre-LN self-attention transformer layer with standard FFN: branches
 * normalize their own input (QNorm inside attention, ln2 before the FFN), the
 * residual stream stays raw.
 */
export function applySelfTransformerLayer(
    {
        name,
        heads,
        token,
        qMask,
        kvMask,
    }: {
        name: string,
        heads: number,
        token: tf.SymbolicTensor;
        qMask?: tf.SymbolicTensor;
        kvMask?: tf.SymbolicTensor;
    },
) {
    const dModel = token.shape[token.shape.length - 1]!;

    const selfAttn = applyCrossAttentionLayer({
        name,
        heads,
        qTok: token,
        qMask,
        kvTok: token,
        kvMask,
        preNorm: true,
    });

    const attnResidual = tf.layers.add({name: `${name}_residual`})
        .apply([token, selfAttn]) as tf.SymbolicTensor;

    const ffnNorm = createNormalizationLayer({
        name: `${name}_ln2`,
    }).apply(attnResidual) as tf.SymbolicTensor;

    // SiLU-gated FFN: gate(x) * linear(x)
    const ffnGate = createDenseLayer({
        name: `${name}_ffn_gate`,
        units: dModel * 4,
        useBias: false,
        activation: 'sigmoid',
    }).apply(ffnNorm) as tf.SymbolicTensor;

    const ffnUp = createDenseLayer({
        name: `${name}_ffn1`,
        units: dModel * 4,
        useBias: false,
        activation: 'linear',
    }).apply(ffnNorm) as tf.SymbolicTensor;

    const ffnInner = tf.layers.multiply({name: `${name}_ffn_silu`})
        .apply([ffnUp, ffnGate]) as tf.SymbolicTensor;

    const ffnOut = createDenseLayer({
        name: `${name}_ffn2`,
        units: dModel,
        useBias: false,
        activation: 'linear',
    }).apply(ffnInner) as tf.SymbolicTensor;

    return tf.layers.add({name: `${name}_ffnAdd`})
        .apply([attnResidual, ffnOut]) as tf.SymbolicTensor;
}

/**
 * Apply a modern (GPT-2/LLaMA-style) self-attention encoder: normalize the
 * embedding once before the first layer (raw projected tokens are usually an
 * order below the ~1 the attention score scale assumes), pre-LN blocks, and a
 * final norm on the output (gathers the residual stream growth that pre-LN
 * leaves unnormalized).
 */
export function applySelfTransformLayers(name: string, {
    depth,
    heads,
    token,
    qMask,
    kvMask,
}: {
    depth: number,
    heads: number,
    token: tf.SymbolicTensor | ((name: string, i: number) => tf.SymbolicTensor),
    /** Which tokens get attention output. Omit for "all tokens query". */
    qMask?: tf.SymbolicTensor | ((name: string, i: number) => tf.SymbolicTensor),
    /** Which tokens can be read. Omit for "all tokens readable". */
    kvMask?: tf.SymbolicTensor | ((name: string, i: number) => tf.SymbolicTensor),
}) {
    const resolve = (
        mask: undefined | tf.SymbolicTensor | ((name: string, i: number) => tf.SymbolicTensor),
        lName: string,
        i: number,
    ) => mask ? (typeof mask === 'function' ? mask(lName, i) : mask) : undefined;

    let x = typeof token === 'function' ? token(name, 0) : token;
    x = createNormalizationLayer({ name: `${name}_inputNorm` }).apply(x) as tf.SymbolicTensor;
    for (let i = 0; i < depth; i++) {
        const lName = `${name}/depth${i}`;
        x = applySelfTransformerLayer({
            name: lName,
            heads,
            token: x,
            qMask: resolve(qMask, lName, i),
            kvMask: resolve(kvMask, lName, i),
        });
    }

    return createNormalizationLayer({ name: `${name}_outputNorm` }).apply(x) as tf.SymbolicTensor;
}

export function applyGlobalAverage1d({ name }: { name: string }, token: tf.SymbolicTensor) {
    return tf.layers.globalAveragePooling1d({ name: name + '_GlobalAvgPool1D' })
        .apply(token) as tf.SymbolicTensor;
}