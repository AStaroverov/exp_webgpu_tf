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

export function tokenProj(x: tf.SymbolicTensor, dModel: number, name: string): SymbolicTensor {
    return tf.layers.dense({ units: dModel, useBias: false, name: name + '_tokProj' }).apply(x) as SymbolicTensor;
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
        name: name + '_CrossAttentionQNorm_' + qTok.name,
        epsilon: 1e-5,
    }).apply(qTok) as tf.SymbolicTensor;

    const attentionInputs = [qTokNorm, kvTok];
    kvMask && attentionInputs.push(kvMask);

    const attention = new MultiHeadAttentionLayer({
        name: name + '_CrossAttentionLayer',
        keyDim: dModel / numHeads,
        numHeads: numHeads,
    }).apply(attentionInputs) as tf.SymbolicTensor;

    const output = tf.layers.dense({
        name: name + '_CrossAttentionOutput',
        units: dModel,
        useBias: true,
    }).apply(attention) as SymbolicTensor;

    return output;
}

export function applySelfAttentionLayer(
    name: string,
    numHeads: number,
    {
        tokens,
        mask,
    }: {
        tokens: tf.SymbolicTensor,
        mask?: tf.SymbolicTensor,
    },
) {
    const dModel = tokens.shape[tokens.shape.length - 1]!;
    const tokensNorm = tf.layers.layerNormalization({
        name: name + '_SelfAttentionTokensNorm_' + tokens.name,
        epsilon: 1e-5,
    }).apply(tokens) as tf.SymbolicTensor;

    const attentionInputs = [tokensNorm, tokensNorm];
    mask && attentionInputs.push(mask);

    const attention = new MultiHeadAttentionLayer({
        name: name + '_SelfAttentionLayer',
        keyDim: dModel / numHeads,
        numHeads: numHeads,
    }).apply(attentionInputs) as tf.SymbolicTensor;

    const attentionOutput = tf.layers.dense({
        name: name + '_SelfAttentionOutput',
        units: dModel,
        useBias: true,
    }).apply(attention) as SymbolicTensor;

    let output = tf.layers.add({ name: name + '_SelfAttentionResidual' })
        .apply([attentionOutput, tokens]) as tf.SymbolicTensor;

    return output;
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
