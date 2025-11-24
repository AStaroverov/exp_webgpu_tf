import * as tf from '@tensorflow/tfjs';
import { SymbolicTensor } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { DenseLayerArgs } from '@tensorflow/tfjs-layers/dist/layers/core';
import {
    ALLY_FEATURES_DIM,
    ALLY_SLOTS, BULLET_FEATURES_DIM,
    BULLET_SLOTS, ENEMY_FEATURES_DIM,
    ENEMY_SLOTS,
    TANK_FEATURES_DIM
} from './Create.ts';
import { MaskLikeLayer } from './Layers/MaskLikeLayer.ts';
import { MultiHeadAttentionLayer } from './Layers/MultiHeadAttentionLayer.ts';
import { RMSNormConfig, RMSNormLayer } from "./Layers/RMSNormLayer.ts";
import { VariableLayer } from './Layers/VariableLayer.ts';

export function createInputs(name: string) {
    const tankInput = tf.input({name: name + '_tankInput', shape: [TANK_FEATURES_DIM]});
    const enemiesInput = tf.input({name: name + '_enemiesInput', shape: [ENEMY_SLOTS, ENEMY_FEATURES_DIM]});
    const enemiesMaskInput = tf.input({name: name + '_enemiesMaskInput', shape: [ENEMY_SLOTS]});
    const alliesInput = tf.input({name: name + '_alliesInput', shape: [ALLY_SLOTS, ALLY_FEATURES_DIM]});
    const alliesMaskInput = tf.input({name: name + '_alliesMaskInput', shape: [ALLY_SLOTS]});
    const bulletsInput = tf.input({name: name + '_bulletsInput', shape: [BULLET_SLOTS, BULLET_FEATURES_DIM]});
    const bulletsMaskInput = tf.input({name: name + '_bulletsMaskInput', shape: [BULLET_SLOTS]});

    return {
        tankInput,
        enemiesInput,
        enemiesMaskInput,
        alliesInput,
        alliesMaskInput,
        bulletsInput,
        bulletsMaskInput,
    };
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

export function createDenseLayer(options: DenseLayerArgs & Required<Pick<DenseLayerArgs, 'useBias' | 'activation'>>) {
    return tf.layers.dense(options)
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

export function convertInputsToTokens(
    {
        tankInput,
        enemiesInput,
        alliesInput,
        bulletsInput,
    }: ReturnType<typeof createInputs>,
    dModel: number,
) {
    const addTypeEmbedding = (x: tf.SymbolicTensor) => {
        const typeEmb = new VariableLayer({
            name: `${x.name}_typeEmbedding`,
            shape: [dModel],
        }).apply(x) as tf.SymbolicTensor;
        return tf.layers.add({ name: `${x.name}_withTypeEmbedding` }).apply([x, typeEmb]) as tf.SymbolicTensor;
    }
    const reshape = (x: tf.SymbolicTensor) => {
        return tf.layers.reshape({
            name: `${x.name}_reshape`,
            targetShape: [1, dModel],
        }).apply(x) as tf.SymbolicTensor;
    };

    const tankTok = reshape(addTypeEmbedding(tokenProj(tankInput, dModel, tankInput.name)));
    const alliesTok = addTypeEmbedding(tokenProj(alliesInput, dModel, alliesInput.name));
    const enemiesTok = addTypeEmbedding(tokenProj(enemiesInput, dModel, enemiesInput.name));
    const bulletsTok = addTypeEmbedding(tokenProj(bulletsInput, dModel, bulletsInput.name));

    return {
        tankTok,
        alliesTok,
        enemiesTok,
        bulletsTok,
    };
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
    qTok = preNorm
        ? createNormalizationLayer({ name: name + '_QNorm_' + qTok.name}).apply(qTok) as tf.SymbolicTensor
        : qTok;
    kvTok = qTok === kvTok
        ? qTok
        : preNorm
            ? createNormalizationLayer({ name: name + '_KVNorm_' + kvTok.name}).apply(kvTok) as tf.SymbolicTensor
            : kvTok;

    // Create mask-like layers if masks are not provided
    qMask ??= new MaskLikeLayer({ name: qTok.name + '_qMaskLike' }).apply(qTok) as tf.SymbolicTensor;
    kvMask ??= new MaskLikeLayer({ name: kvTok.name + '_kvMaskLike' }).apply(kvTok) as tf.SymbolicTensor;

    const attention = new MultiHeadAttentionLayer({
        name: name + '_MultiHeadAttentionLayer',
        keyDim: dModel / heads,
        numHeads: heads,
    }).apply([qTok, qMask, kvTok, kvMask]) as tf.SymbolicTensor;

    return attention;
}

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

export function applyCrossTransformerLayers({
    name,
    depth,
    heads,
    qTok,
    kvTok,
    qMask,
    kvMask,
    preNorm = false,
}: {
    name: string, 
    depth: number,
    heads: number,
    qTok: tf.SymbolicTensor | (() => tf.SymbolicTensor),
    kvTok: tf.SymbolicTensor | (() => tf.SymbolicTensor),
    qMask?: tf.SymbolicTensor | (() => tf.SymbolicTensor),
    kvMask?: tf.SymbolicTensor | (() => tf.SymbolicTensor),
    preNorm?: boolean,
}) {
    let x = typeof qTok === 'function' ? qTok() : qTok;
    for (let i = 0; i < depth; i++) {
        x = applyCrossAttentionLayer({
            name: `${name}/depth${i}`,
            heads,
            qTok: x,
            kvTok: typeof kvTok === 'function' ? kvTok() : kvTok,
            qMask: typeof qMask === 'function' ? qMask() : qMask,
            kvMask: typeof kvMask === 'function' ? kvMask() : kvMask,
            preNorm,
        });
    }

    return x;
}

export function applySelfTransformerLayer(
    {
        name,
        heads,
        token,
        mask,
        preNorm = false,
    }: {
        name: string,
        heads: number,
        token: tf.SymbolicTensor;
        mask?: tf.SymbolicTensor;
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

export function applySelfTransformLayers(name: string, {
    depth,
    heads,
    token,
    mask,
    preNorm = false,
}: {
    depth: number,
    heads: number,
    token: tf.SymbolicTensor | ((i: number) => tf.SymbolicTensor),
    mask?: tf.SymbolicTensor | ((i: number) => tf.SymbolicTensor),
    preNorm?: boolean,
}) {
    let x = typeof token === 'function' ? token(0) : token;
    for (let i = 0; i < depth; i++) {
        x = applySelfTransformerLayer({
            name: `${name}/depth${i}`,
            heads,
            token: x,
            mask: mask ? (typeof mask === 'function' ? mask(i) : mask) : undefined,
            preNorm,
        });
    }

    return x;
}

export function applyGlobalAverage1d({ name }: { name: string }, token: tf.SymbolicTensor) {
    return tf.layers.globalAveragePooling1d({ name: name + '_GlobalAvgPool1D' })
        .apply(token) as tf.SymbolicTensor;
}