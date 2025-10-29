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
    finalMLP: [ActivationIdentifier, number][];
};

export function createNetwork(modelName: Model, config: NetworkConfig) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, config.dim);

    const tankToBattleAttn = attentionToTank({
        name: modelName + '_tankToBattleAttn',
        heads: config.heads,
        qTok: tokens.tankTok,
        kvTok: tokens.battleTok,
    });
    const tankToEnemiesAttn = attentionToTank({
        name: modelName + '_tankToEnemiesAttn',
        heads: config.heads,
        qTok: tokens.tankTok,
        kvTok: tokens.enemiesTok,
        kvMask: inputs.enemiesMaskInput,
    });
    const tankToAlliesAttn = attentionToTank({
        name: modelName + '_tankToAlliesAttn',
        heads: config.heads,
        qTok: tokens.tankTok,
        kvTok: tokens.alliesTok,
        kvMask: inputs.alliesMaskInput,
    });
    const tankToBulletsAttn = attentionToTank({
        name: modelName + '_tankToBulletsAttn',
        heads: config.heads,
        qTok: tokens.tankTok,
        kvTok: tokens.bulletsTok,
        kvMask: inputs.bulletsMaskInput,
    });

    const tankContextToken = tf.layers.concatenate({ name: modelName + '_tankContextToken', axis: 1 }).apply([
        tokens.tankTok,
        tankToBattleAttn,
        tankToEnemiesAttn,
        tankToAlliesAttn,
        tankToBulletsAttn,
    ]) as tf.SymbolicTensor;

    const transformedTankContextToken = applySelfTransformLayers(
        modelName + '_transformedTankContextToken',
        {
            token: tankContextToken,
            depth: 3,
            heads: config.heads,
        },
    );

    const finalToken = tf.layers.concatenate({ name: modelName + '_finalToken', axis: 1 }).apply([
        tokens.controllerTok,
        tokens.tankTok,
        transformedTankContextToken,
    ]) as tf.SymbolicTensor;

    const flattenedFinalToken = tf.layers.flatten({ name: modelName + '_flattenedFinalToken' }).apply(finalToken) as tf.SymbolicTensor;
    const normFinalToken = createNormalizationLayer({ name: modelName + '_normFinalToken' }).apply(flattenedFinalToken) as tf.SymbolicTensor;

    const finalMLP = applyMLP(
        modelName + '_finalMLP',
        normFinalToken,
        config.finalMLP,
    );

    return { inputs, network: finalMLP, phi: normFinalToken };
}


function attentionToTank(config: {
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