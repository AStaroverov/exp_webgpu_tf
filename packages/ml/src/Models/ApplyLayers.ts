import * as tf from '@tensorflow/tfjs';
import { SymbolicTensor } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { DenseLayerArgs } from '@tensorflow/tfjs-layers/dist/layers/core';
import {
    ALLY_FEATURES_DIM,
    ALLY_SLOTS,
    BATTLE_FEATURES_DIM,
    BULLET_FEATURES_DIM,
    BULLET_SLOTS,
    CONTROLLER_FEATURES_DIM,
    ENEMY_FEATURES_DIM,
    ENEMY_SLOTS,
    TANK_FEATURES_DIM,
} from './Create.ts';
import { AttentionPoolLayer } from './Layers/AttentionPoolLayer.ts';
import { MultiHeadAttentionLayer } from './Layers/MultiHeadAttentionLayer.ts';
import { RMSNormConfig, RMSNormLayer } from "./Layers/RMSNormLayer.ts";

export function createInputs(name: string) {
    const controllerInput = tf.input({name: name + '_controllerInput', shape: [CONTROLLER_FEATURES_DIM]});
    const battleInput = tf.input({name: name + '_battlefieldInput', shape: [BATTLE_FEATURES_DIM]});
    const tankInput = tf.input({name: name + '_tankInput', shape: [TANK_FEATURES_DIM]});

    const enemiesInput = tf.input({name: name + '_enemiesInput', shape: [ENEMY_SLOTS, ENEMY_FEATURES_DIM]});
    const enemiesMaskInput = tf.input({name: name + '_enemiesMaskInput', shape: [ENEMY_SLOTS]});
    const alliesInput = tf.input({name: name + '_alliesInput', shape: [ALLY_SLOTS, ALLY_FEATURES_DIM]});
    const alliesMaskInput = tf.input({name: name + '_alliesMaskInput', shape: [ALLY_SLOTS]});
    const bulletsInput = tf.input({name: name + '_bulletsInput', shape: [BULLET_SLOTS, BULLET_FEATURES_DIM]});
    const bulletsMaskInput = tf.input({name: name + '_bulletsMaskInput', shape: [BULLET_SLOTS]});

    return {
        controllerInput,
        battleInput,
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

export function applySelfTransformLayers(name: string, {
    depth,
    heads,
    token,
    mask,
    preNorm = false,
}: {
    depth: number,
    heads: number,
    token: tf.SymbolicTensor,
    mask?: tf.SymbolicTensor,
    preNorm?: boolean,
}) {
    let x = token;
    for (let i = 0; i < depth; i++) {
        x = applySelfTransformerLayer({
            name: `${name}/transformer${i}`,
            heads,
            token: x,
            mask,
            preNorm,
        });
    }

    return x;
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
        controllerInput,
        battleInput,
        tankInput,
        enemiesInput,
        alliesInput,
        bulletsInput,
    }: ReturnType<typeof createInputs>,
    dModel: number,
) {
    const reshape = (x: tf.SymbolicTensor) => {
        return tf.layers.reshape({
            name: `${x.name}_reshape`,
            targetShape: [1, dModel],
        }).apply(x) as tf.SymbolicTensor;
    };
    const controllerTok = reshape(tokenProj(controllerInput, dModel, controllerInput.name));
    const battleTok = reshape(tokenProj(battleInput, dModel, battleInput.name));
    const tankTok = reshape(tokenProj(tankInput, dModel, tankInput.name));
    const alliesTok = tokenProj(alliesInput, dModel, alliesInput.name);
    const enemiesTok = tokenProj(enemiesInput, dModel, enemiesInput.name);
    const bulletsTok = tokenProj(bulletsInput, dModel, bulletsInput.name);

    return {
        controllerTok,
        battleTok,
        tankTok,
        alliesTok,
        enemiesTok,
        bulletsTok,
    };
}

export function applyCrossAttentionLayer(
    {
        name,
        heads: heads,
        qTok,
        kvTok,
        kvMask,
        preNorm = false,
    }: {
        name: string,
        heads: number,
        qTok: tf.SymbolicTensor,
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

    const attentionInputs = [qTok, kvTok];
    kvMask && attentionInputs.push(kvMask);

    const attention = new MultiHeadAttentionLayer({
        name: name + '_MultiHeadAttentionLayer',
        keyDim: dModel / heads,
        numHeads: heads,
    }).apply(attentionInputs) as tf.SymbolicTensor;

    const output = createDenseLayer({
        name: name + '_output',
        units: dModel,
        useBias: false,
        activation: 'linear',
    }).apply(attention) as SymbolicTensor;

    return output;
}

export function applyPoolAttentionLayer(
    {
        name,
        heads,
        token,
        mask,
        preNorm = false,
    }: {
        name: string,
        heads: number,
        token: tf.SymbolicTensor,
        mask?: tf.SymbolicTensor,
        preNorm?: boolean,
    },
) {
    const dModel = token.shape[token.shape.length - 1]!;
    
    token = preNorm
        ? createNormalizationLayer({ name: name + '_Norm_' + token.name }).apply(token) as tf.SymbolicTensor
        : token;

    const inputs = [token];
    mask && inputs.push(mask);

    const attention = new AttentionPoolLayer({
        name: name + '_AttentionPoolLayer',
        keyDim: dModel / heads,
        numHeads: heads,
    }).apply(inputs) as tf.SymbolicTensor;

    const output = createDenseLayer({
        name: name + '_output',
        units: dModel,
        useBias: false,
        activation: 'linear',
    }).apply(attention) as SymbolicTensor;

    return output;
}

export function applyCrossTransformerLayer(
    {
        name,
        heads,
        qTok,
        kvTok,
        kvMask,
        preNorm = false,
    }: {
        name: string,
        heads: number,
        qTok: tf.SymbolicTensor,
        kvTok: tf.SymbolicTensor,
        kvMask?: tf.SymbolicTensor,
        preNorm?: boolean,
    },
) {
    const dModel = qTok.shape[qTok.shape.length - 1]!;

    const crossAttn = applyCrossAttentionLayer({name, heads,qTok, kvTok, kvMask, preNorm});

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
    const selfAttn = applyCrossAttentionLayer({name, heads, qTok: token, kvTok: token, kvMask: mask, preNorm});

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
