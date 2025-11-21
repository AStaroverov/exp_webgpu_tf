import * as tf from '@tensorflow/tfjs';
import {
    applyCrossTransformerLayer,
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
    dim: 128,
    heads: 4,
    depth: 4,
    headsMLP: [0, 1, 2, 3].map(() =>
        ([
            ['relu', 128] as const,
            ['relu', 64] as const,
        ])
    )
};

const valueNetworkConfig: NetworkConfig = {
    dim: 64,
    heads: 1,
    depth: 2,
    headsMLP: [[
        ['relu', 64],
        ['relu', 32],
    ]],
};

export function createNetwork(modelName: Model, config: NetworkConfig = modelName === Model.Policy ? policyNetworkConfig : valueNetworkConfig) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, config.dim);

    const enemiesAttn = getAttention({
        name: modelName + '_tankToEnemiesAttn',
        heads: config.heads,
        qTok: tokens.tankTok,
        kvTok: tokens.enemiesTok,
        kvMask: inputs.enemiesMaskInput,
    });
    const alliesAttn = getAttention({
        name: modelName + '_tankToAlliesAttn',
        heads: config.heads,
        qTok: tokens.tankTok,
        kvTok: tokens.alliesTok,
        kvMask: inputs.alliesMaskInput,
    });
    const bulletsAttn = getAttention({
        name: modelName + '_tankToBulletsAttn',
        heads: config.heads,
        qTok: tokens.tankTok,
        kvTok: tokens.bulletsTok,
        kvMask: inputs.bulletsMaskInput,
    });

    const contextToken = tf.layers.concatenate({name: modelName + '_tankContextToken', axis: 1 }).apply([
        tokens.battleTok,
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

    const attentionToController = applyCrossTransformerLayer({
        name: modelName + '_contextToControllerAttn',
        heads: config.heads,
        qTok: tokens.controllerTok,
        kvTok: transformedContextToken,
        // preNorm: true,
    });

    // const flattenedFinalToken = tf.layers.flatten({name: modelName + '_flattenedTankContextToken'})
    //     .apply(transformedContextToken) as tf.SymbolicTensor;

    // const finalMLP = applyMLP({
    //     name: modelName + '_finalMLP',
    //     layers: [
    //         ['relu', config.dim * 8],
    //         ['relu', config.dim * 6],
    //         ['relu', config.dim * 4],
    //         ['relu', config.dim * 2],
    //     ],
    //     preNorm: true,
    // }, flattenedFinalToken);

    const flattenedAttentionToController = tf.layers.flatten({name: modelName + '_flattenedAttentionToController'})
        .apply(attentionToController) as tf.SymbolicTensor;

    const heads = config.headsMLP.map((layers, i) => {
        return applyMLP({
            name: modelName + '_headsMLP_' + i,
            layers,
        }, flattenedAttentionToController);
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