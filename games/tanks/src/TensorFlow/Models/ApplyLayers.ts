import * as tf from '@tensorflow/tfjs';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { SymbolicTensor } from '@tensorflow/tfjs-layers/dist/engine/topology';
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
import { FixedPositionalEncodingLayer } from './Layers/FixedPositionalEncodingLayer.ts';
import { RoleEmbeddingLayer } from './Layers/RoleEncodingLayer.ts';
import { AttentionPoolLayer } from './Layers/AttentionPoolLayer.ts';

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

export function applyDenseLayers(name: string, layer: tf.SymbolicTensor, hiddenLayers: [ActivationIdentifier, number][]) {
    let x = layer;
    let i = 0;
    for (const [activation, units] of hiddenLayers) {
        x = tf.layers.dense({
            name: `${ name }/dense${ i++ }`,
            units,
            activation,
            kernelInitializer: 'glorotUniform',
        }).apply(x) as tf.SymbolicTensor;
    }

    return x;
}


export function applyTransformLayers(name: string, {
    depth,
    numHeads,
    dropout,
    token,
    mask,
}: {
    depth: number,
    numHeads: number,
    dropout?: number;
    token: tf.SymbolicTensor,
    mask?: tf.SymbolicTensor
}) {
    let x = token;
    for (let i = 0; i < depth; i++) {
        x = applyTransformerLayer(`${ name }/transformer${ i }`, {
            numHeads,
            dropout,
            token: x,
            mask,
        });
    }

    return x;
}

export function tokenProj(x: tf.SymbolicTensor, dModel: number, name: string): SymbolicTensor {
    return tf.layers.dense({ units: dModel, useBias: true, name: name + '_tokProj' }).apply(x) as SymbolicTensor;
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
            name: `${ x.name }_reshape`,
            targetShape: [1, dModel],
        }).apply(x) as tf.SymbolicTensor;
    };

    const controllerTok =
        applyEncoding(
            reshape(tokenProj(controllerInput, dModel, controllerInput.name)));
    const battleTok =
        applyEncoding(
            reshape(tokenProj(battleInput, dModel, battleInput.name)));
    const tankTok =
        applyEncoding(
            reshape(tokenProj(tankInput, dModel, tankInput.name)));
    const alliesTok =
        applyEncoding(
            tokenProj(alliesInput, dModel, alliesInput.name));
    const enemiesTok =
        applyEncoding(
            tokenProj(enemiesInput, dModel, enemiesInput.name));
    const bulletsTok =
        applyEncoding(
            tokenProj(bulletsInput, dModel, bulletsInput.name));

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
    const qTokNorm = tf.layers.layerNormalization({
        name: name + '_QNorm_' + qTok.name,
        epsilon: 1e-5,
    }).apply(qTok) as tf.SymbolicTensor;

    const attentionInputs = [qTokNorm, kvTok];
    kvMask && attentionInputs.push(kvMask);

    const attention = new MultiHeadAttentionLayer({
        name: name + '_MultiHeadAttentionLayer',
        keyDim: dModel / numHeads,
        numHeads: numHeads,
    }).apply(attentionInputs) as tf.SymbolicTensor;

    const output = tf.layers.dense({
        name: name + '_output',
        units: dModel,
        useBias: true,
    }).apply(attention) as SymbolicTensor;

    return output;
}

export function applySelfAttentionLayer(
    name: string,
    numHeads: number,
    {
        token,
        mask,
    }: {
        token: tf.SymbolicTensor,
        mask?: tf.SymbolicTensor,
    },
) {
    const dModel = token.shape[token.shape.length - 1]!;
    const tokensNorm = tf.layers.layerNormalization({
        name: name + '_QNorm_' + token.name,
        epsilon: 1e-5,
    }).apply(token) as tf.SymbolicTensor;

    const attentionInputs = [tokensNorm, tokensNorm];
    mask && attentionInputs.push(mask);

    const attention = new MultiHeadAttentionLayer({
        name: name + '_MultiHeadAttentionLayer',
        keyDim: dModel / numHeads,
        numHeads: numHeads,
    }).apply(attentionInputs) as tf.SymbolicTensor;

    const output = tf.layers.dense({
        name: name + '_output',
        units: dModel,
        useBias: true,
    }).apply(attention) as SymbolicTensor;

    return output;
}

export function applyTransformerLayer(
    name: string,
    {
        numHeads,
        dropout,
        token,
        mask,
    }: {
        numHeads: number,
        dropout?: number;
        token: tf.SymbolicTensor;
        mask?: tf.SymbolicTensor;
    },
) {
    const dModel = token.shape[token.shape.length - 1]!;
    const selfAttn = applySelfAttentionLayer(name, numHeads, { token, mask });

    const attnProj = dropout == null ? selfAttn : tf.layers.dropout({ rate: dropout, name: `${ name }_drop` })
        .apply(selfAttn) as tf.SymbolicTensor;

    const attnResidual = tf.layers.add({ name: `${ name }_residual` })
        .apply([token, attnProj]) as tf.SymbolicTensor;

    const norm2 = tf.layers.layerNormalization({
        name: `${ name }_ln2`,
        epsilon: 1e-5,
    }).apply(attnResidual) as tf.SymbolicTensor;

    const ffnInner = tf.layers.dense({
        name: `${ name }_ffn1`,
        units: dModel * 4,
        useBias: true,
        activation: 'relu',
    }).apply(norm2) as tf.SymbolicTensor;

    const ffnOut = tf.layers.dense({
        name: `${ name }_ffn2`,
        units: dModel,
        useBias: true,
    }).apply(ffnInner) as tf.SymbolicTensor;

    const ffnDrop = dropout == null ? ffnOut : tf.layers.dropout({ rate: dropout, name: `${ name }_ffnDrop` })
        .apply(ffnOut) as tf.SymbolicTensor;

    const finalOut = tf.layers.add({ name: `${ name }_ffnAdd` })
        .apply([attnResidual, ffnDrop]) as tf.SymbolicTensor;

    return finalOut;
}

export function applyAttentionPool(
    name: string,
    tokens: tf.SymbolicTensor,
    dropout?: number,
) {
    let x = new AttentionPoolLayer({ name: name + '_AttentionPoolLayer' }).apply(tokens) as tf.SymbolicTensor;
    x = tf.layers.layerNormalization({ name: name + '_norm' }).apply(x) as tf.SymbolicTensor;
    x = dropout == null ? x : tf.layers.dropout({
        name: name + '_dropout',
        rate: dropout,
    }).apply(x) as tf.SymbolicTensor;

    return x;
}

export function applyEncoding(token: tf.SymbolicTensor): tf.SymbolicTensor {
    const N = token.shape[1]!;

    const posEmbedding = N === 1
        ? token
        : new FixedPositionalEncodingLayer({ name: token.name + '_withPos' })
            .apply(token) as tf.SymbolicTensor;

    const roleEmbedding = new RoleEmbeddingLayer({ name: posEmbedding.name + 'withRole' })
        .apply(posEmbedding) as tf.SymbolicTensor;

    return roleEmbedding;
}
