import * as tf from '@tensorflow/tfjs';
import { SymbolicTensor } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { DenseLayerArgs } from '@tensorflow/tfjs-layers/dist/layers/core';
import { LayerNormalizationLayerArgs } from '@tensorflow/tfjs-layers/dist/layers/normalization';
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
import { MultiHeadAttentionLayer } from './Layers/MultiHeadAttentionLayer.ts';

export function createInputs(name: string) {
    const controllerInput = tf.input({ name: name + '_controllerInput', shape: [CONTROLLER_FEATURES_DIM] });
    const battleInput = tf.input({ name: name + '_battlefieldInput', shape: [BATTLE_FEATURES_DIM] });
    const tankInput = tf.input({ name: name + '_tankInput', shape: [TANK_FEATURES_DIM] });

    const enemiesInput = tf.input({ name: name + '_enemiesInput', shape: [ENEMY_SLOTS, ENEMY_FEATURES_DIM] });
    const enemiesMaskInput = tf.input({ name: name + '_enemiesMaskInput', shape: [ENEMY_SLOTS] });
    const alliesInput = tf.input({ name: name + '_alliesInput', shape: [ALLY_SLOTS, ALLY_FEATURES_DIM] });
    const alliesMaskInput = tf.input({ name: name + '_alliesMaskInput', shape: [ALLY_SLOTS] });
    const bulletsInput = tf.input({ name: name + '_bulletsInput', shape: [BULLET_SLOTS, BULLET_FEATURES_DIM] });
    const bulletsMaskInput = tf.input({ name: name + '_bulletsMaskInput', shape: [BULLET_SLOTS] });

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

export function applyMLP(name: string, layer: tf.SymbolicTensor, hiddenLayers: [ActivationIdentifier, number][]) {
    let x = layer;
    let i = 0;
    for (const [activation, units] of hiddenLayers) {
        x = createDenseLayer({ name: `${name}/dense${i++}`, units, activation, useBias: true }).apply(x) as tf.SymbolicTensor;
    }

    return x;
}

export function createDenseLayer(options: DenseLayerArgs & Required<Pick<DenseLayerArgs, 'useBias' | 'activation'>>) {
    return tf.layers.dense(options)
}

export function createNormalizationLayer(options: LayerNormalizationLayerArgs) {
    return tf.layers.layerNormalization(options);
}

export function applySelfTransformLayers(name: string, {
    depth,
    heads: numHeads,
    token,
    mask,
}: {
    depth: number,
    heads: number,
    token: tf.SymbolicTensor,
    mask?: tf.SymbolicTensor
}) {
    let x = token;
    for (let i = 0; i < depth; i++) {
        x = applySelfTransformerLayer(`${name}/transformer${i}`, {
            numHeads,
            token: x,
            mask,
        });
    }

    return x;
}

export function tokenProj(x: tf.SymbolicTensor, dModel: number, name: string): SymbolicTensor {
    return createDenseLayer({ name: name + '_tokProj', units: dModel, useBias: false, activation: 'linear' }).apply(x) as SymbolicTensor;
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

export function applyAttentionPoolingLayer(
    name: string,
    numHeads: number,
    {
        qTok,
        kvTok,
        kvMask,
    }: {
        qTok: tf.SymbolicTensor,
        kvTok: tf.SymbolicTensor,
        kvMask?: tf.SymbolicTensor,
    },
) {
    const dModel = qTok.shape[qTok.shape.length - 1]!;
    const qTokNorm = createNormalizationLayer({
        name: name + '_QNorm_' + qTok.name,
    }).apply(qTok) as tf.SymbolicTensor;
    const kvTokNorm = createNormalizationLayer({
        name: name + '_KVNorm_' + kvTok.name,
    }).apply(kvTok) as tf.SymbolicTensor;

    const attentionInputs = [qTokNorm, kvTokNorm];
    kvMask && attentionInputs.push(kvMask);

    const attention = new MultiHeadAttentionLayer({
        name: name + '_MultiHeadAttentionLayer',
        keyDim: dModel / numHeads,
        numHeads: numHeads,
    }).apply(attentionInputs) as tf.SymbolicTensor;

    const output = createDenseLayer({
        name: name + '_output',
        units: dModel,
        useBias: false,
        activation: 'linear',
    }).apply(attention) as SymbolicTensor;

    return output;
}

export function applyCrossAttentionLayer(
    name: string,
    {
        heads: numHeads,
        qTok,
        kvTok,
        kvMask,
    }: {
        heads: number,
        qTok: tf.SymbolicTensor,
        kvTok: tf.SymbolicTensor,
        kvMask?: tf.SymbolicTensor
    },
) {
    const dModel = qTok.shape[qTok.shape.length - 1]!;

    const crossAttn = applyAttentionPoolingLayer(name, numHeads, { qTok, kvTok, kvMask });

    const attnResidual = tf.layers.add({ name: `${name}_residual` })
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

    const finalOut = tf.layers.add({ name: `${name}_ffnAdd` })
        .apply([attnResidual, ffnOut]) as tf.SymbolicTensor;

    return finalOut;
}

export function applySelfTransformerLayer(
    name: string,
    {
        numHeads,
        token,
        mask,
    }: {
        numHeads: number,
        token: tf.SymbolicTensor;
        mask?: tf.SymbolicTensor;
    },
) {
    const dModel = token.shape[token.shape.length - 1]!;
    const selfAttn = applyAttentionPoolingLayer(name, numHeads, { qTok: token, kvTok: token, kvMask: mask });

    const attnResidual = tf.layers.add({ name: `${name}_residual` })
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

    const finalOut = tf.layers.add({ name: `${name}_ffnAdd` })
        .apply([attnResidual, ffnOut]) as tf.SymbolicTensor;

    return finalOut;
}
