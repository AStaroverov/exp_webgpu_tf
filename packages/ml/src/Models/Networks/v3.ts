import * as tf from '@tensorflow/tfjs';
import {
    applyCrossAttentionLayer,
    applyMLP,
    applySelfTransformLayers,
    convertInputsToTokens,
    createInputs,
    createNormalizationLayer
} from '../ApplyLayers.ts';

import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { Model } from '../def.ts';

type NetworkConfig = {
    dim: number;
    heads: number;
    depth: number;
    MLP: [ActivationIdentifier, number][];
};

type policyNetworkConfig = NetworkConfig

const policyNetworkConfig: policyNetworkConfig = {
    dim: 64,
    heads: 4,
    depth: 3,
    MLP: [
        ['relu', 512],
        ['relu', 512],
        ['relu', 256],
        ['relu', 128],
    ],
};

const valueNetworkConfig: NetworkConfig = {
    dim: 32,
    heads: 2,
    depth: 2,
    MLP: [
        ['relu', 128],
        ['relu', 64],
        ['relu', 32],
    ] as [ActivationIdentifier, number][],
};

export function createNetwork(modelName: Model, config: NetworkConfig = modelName === Model.Policy ? policyNetworkConfig : valueNetworkConfig) {
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

    const tankContextToken = tf.layers.concatenate({ name: modelName + '_tankContextToken', axis: 1 }).apply([
        tankToEnemiesAttn,
        tankToAlliesAttn,
        tankToBulletsAttn,
    ]) as tf.SymbolicTensor;

    const transformedTankContextToken = applySelfTransformLayers(
        modelName + '_transformedTankContextToken',
        {
            token: tankContextToken,
            heads: config.heads,
            depth: config.depth,
        },
    );

    const finalToken = tf.layers.concatenate({ name: modelName + '_finalToken', axis: 1 }).apply([
        tokens.tankTok,
        tokens.battleTok,
        tokens.controllerTok,
        transformedTankContextToken,
    ]) as tf.SymbolicTensor;

    const flattenedFinalToken = tf.layers.flatten({ name: modelName + '_flattenedFinalToken' }).apply(finalToken) as tf.SymbolicTensor;

    const finalMLP = applyMLP(
        modelName + '_finalMLP',
        flattenedFinalToken,
        config.MLP,
    );

    return { inputs, network: finalMLP };
}


function getAttention(config: {
    name: string,
    heads: number,
    qTok: tf.SymbolicTensor,
    kvTok: tf.SymbolicTensor,
    kvMask?: tf.SymbolicTensor
}) {
    const tankAttn = applyCrossAttentionLayer(config.name + '_crossAttn', {
        heads: config.heads,
        qTok: config.qTok,
        kvTok: config.kvTok,
        kvMask: config.kvMask,
    });
    const normTankAttn = createNormalizationLayer({ name: config.name + '_norm' }).apply(tankAttn) as tf.SymbolicTensor;

    return normTankAttn;
}