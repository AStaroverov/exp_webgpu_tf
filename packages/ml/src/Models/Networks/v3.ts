import * as tf from '@tensorflow/tfjs';
import {
    applyCrossTransformerLayer,
    applyGlobalAverage1d,
    applyMLP,
    applySelfTransformLayers,
    convertInputsToTokens,
    createInputs
} from '../ApplyLayers.ts';

import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { Model } from '../def.ts';

type NetworkConfig = {
    dim: number;
    heads: number;
    depth: number;
    headsMLP: ([ActivationIdentifier, number][])[];
};

type policyNetworkConfig = NetworkConfig

const policyNetworkConfig: policyNetworkConfig = {
    dim: 32,
    heads: 4,
    depth: 8,
    headsMLP: [0, 1, 2, 3].map(() =>
        ([
            ['relu', 32*4] as const,
            ['relu', 32*4] as const,
            ['relu', 32*4] as const,
        ])
    )
};

const valueNetworkConfig: NetworkConfig = {
    dim: 16,
    heads: 1,
    depth: 3,
    headsMLP: [[
        ['relu', 16*4],
        ['relu', 16*4],
    ]],
};

export function createNetwork(modelName: Model, config: NetworkConfig = modelName === Model.Policy ? policyNetworkConfig : valueNetworkConfig) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, config.dim);

    const enemiesAttn = getAttention({
        name: modelName + '_enemiesAttn',
        heads: config.heads,
        qTok: tokens.tankTok,//getAverage(modelName + '_enemiesAvgPool', tokens.enemiesTok),
        kvTok: tokens.enemiesTok,
        kvMask: inputs.enemiesMaskInput,
    });
    const alliesAttn = getAttention({
        name: modelName + '_alliesAttn',
        heads: config.heads,
        qTok: tokens.tankTok,//getAverage(modelName + '_alliesAvgPool', tokens.alliesTok),
        kvTok: tokens.alliesTok,
        kvMask: inputs.alliesMaskInput,
    });
    const bulletsAttn = getAttention({
        name: modelName + '_bulletsAttn',
        heads: config.heads,
        qTok: tokens.tankTok,//getAverage(modelName + '_bulletsAvgPool', tokens.bulletsTok),
        kvTok: tokens.bulletsTok,
        kvMask: inputs.bulletsMaskInput,
    });

    const contextToken = tf.layers.concatenate({name: modelName + '_tankContextToken', axis: 1 }).apply([
        tokens.tankTok,
        enemiesAttn,
        alliesAttn,
        bulletsAttn,
    ]) as tf.SymbolicTensor;

    const transformedContextToken = applySelfTransformLayers(
        modelName + '_transformedTankContextToken',
        {
            token: contextToken,
            heads: config.heads,
            depth: config.depth,
            preNorm: true,
        },
    );

    const finalToken = tf.layers.concatenate({name: modelName + '_finalToken', axis: 1 }).apply([
        tokens.controllerTok,
        tokens.battleTok,
        transformedContextToken,
    ]) as tf.SymbolicTensor;

    const flattenedFinalToken = tf.layers.flatten({name: modelName + '_flattenedTankContextToken'})
        .apply(finalToken) as tf.SymbolicTensor;

    const finalMLP = applyMLP({
        name: modelName + '_finalMLP',
        layers: [
            ['relu', config.dim * 12],
            ['relu', config.dim * 12],
            ['relu', config.dim * 6],
        ],
        preNorm: true,
    }, flattenedFinalToken);

    
    const heads = config.headsMLP.map((layers, i) => {
        return applyMLP({
            name: modelName + '_headsMLP_' + i,
            layers,
        }, finalMLP);
    })

    return {inputs, heads};
}


function getAttention(config: {
    name: string,
    heads: number,
    qTok: tf.SymbolicTensor,
    kvTok: tf.SymbolicTensor,
    kvMask?: tf.SymbolicTensor
}) {
    return applyCrossTransformerLayer({
        name: config.name + '_crossAttn',
        heads: config.heads,
        qTok: config.qTok,
        kvTok: config.kvTok,
        kvMask: config.kvMask,
    });
}

function getAverage(name: string, token: tf.SymbolicTensor) {
    const average = applyGlobalAverage1d({ name }, token);
    const reshaped = tf.layers.reshape({ targetShape: [1, average.shape[1]!] , name: name + '_reshaped' })
        .apply(average) as tf.SymbolicTensor;
    return reshaped;
}