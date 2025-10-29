import * as tf from '@tensorflow/tfjs';
import {
    applyAttentionPoolingLayer,
    applyMLP,
    convertInputsToTokens,
    createInputs,
    createNormalizationLayer
} from '../ApplyLayers.ts';

import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { Model } from '../def.ts';

type NetworkConfig = {
    dim: number;
    heads: number;
    finalMLP: [ActivationIdentifier, number][];
};

export function createNetwork(modelName: Model, config: NetworkConfig) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, config.dim);

    const tankToEnemiesAttn = getAttention({
        name: modelName + '_tankToEnemiesAttn',
        heads: config.heads,
        qTok: tokens.tankTok,
        kvTok: tokens.enemiesTok,
        kvMask: inputs.enemiesMaskInput,
    });
    const tankToAlliesAttn = getAttention({
        name: modelName + '_tankToAlliesAttn',
        heads: config.heads,
        qTok: tokens.tankTok,
        kvTok: tokens.alliesTok,
        kvMask: inputs.alliesMaskInput,
    });
    const tankToBulletsAttn = getAttention({
        name: modelName + '_tankToBulletsAttn',
        heads: config.heads,
        qTok: tokens.tankTok,
        kvTok: tokens.bulletsTok,
        kvMask: inputs.bulletsMaskInput,
    });

    const finalToken = tf.layers.concatenate({ name: modelName + '_finalToken', axis: 1 }).apply([
        tokens.controllerTok,
        tokens.battleTok,
        tokens.tankTok,
        tankToEnemiesAttn,
        tankToAlliesAttn,
        tankToBulletsAttn,
    ]) as tf.SymbolicTensor;

    const flattenedFinalToken = tf.layers.flatten({ name: modelName + '_flattenedFinalToken' }).apply(finalToken) as tf.SymbolicTensor;

    const finalMLP = applyMLP(
        modelName + '_finalMLP',
        flattenedFinalToken,
        config.finalMLP,
    );

    return { inputs, network: finalMLP, phi: flattenedFinalToken };
}


function getAttention(config: {
    name: string,
    heads: number,
    qTok: tf.SymbolicTensor,
    kvTok: tf.SymbolicTensor,
    kvMask?: tf.SymbolicTensor
}) {
    const tankAttn = applyAttentionPoolingLayer(config.name + '_crossAttn', config.heads, {
        qTok: config.qTok,
        kvTok: config.kvTok,
        kvMask: config.kvMask,
    });
    const normTankAttn = createNormalizationLayer({ name: config.name + '_norm' }).apply(tankAttn) as tf.SymbolicTensor;

    return normTankAttn;
}