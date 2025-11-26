import * as tf from '@tensorflow/tfjs';
import {
    applyCrossTransformerLayers,
    applyGlobalAverage1d,
    applyMLP,
    applySelfTransformLayers,
    convertInputsToTokens,
    createInputs
} from '../ApplyLayers.ts';
import { MaskLikeLayer } from '../Layers/MaskLikeLayer.ts';

import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { Model } from '../def.ts';
import { MaskSquashLayer } from '../Layers/MaskSquashLayer.ts';

type NetworkConfig = {
    dim: number;
    heads: number;
    depth: number;
    headsMLP: ([ActivationIdentifier, number][])[];
};

type policyNetworkConfig = NetworkConfig

const policyNetworkConfig: policyNetworkConfig = {
    dim: 16,
    heads: 1,
    depth: 6,
    headsMLP: [0, 1, 2, 3].map(() =>
        ([
            ['relu', 16*4] as const,
        ])
    )
};

const valueNetworkConfig: NetworkConfig = {
    dim: 16,
    heads: 1,
    depth: 2,
    headsMLP: [[
        ['relu', 16*4],
    ]],
};

export function createNetwork(modelName: Model, config: NetworkConfig = modelName === Model.Policy ? policyNetworkConfig : valueNetworkConfig) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, config.dim);

    const alliesAttn = applyCrossTransformerLayers({
        name: modelName + '_alliesAttn',
        depth: Math.ceil(config.depth / 3),
        heads: config.heads,
        qTok: getAverage(modelName + '_alliesAvgPool', tokens.alliesTok),
        kvTok: tokens.alliesTok,
        kvMask: inputs.alliesMaskInput,
    });

    const enemiesAttn = applyCrossTransformerLayers({
        name: modelName + '_enemiesAttn',
        depth: Math.ceil(config.depth / 3),
        heads: config.heads,
        qTok: getAverage(modelName + '_enemiesAvgPool', tokens.enemiesTok),
        kvTok: tokens.enemiesTok,
        kvMask: inputs.enemiesMaskInput,
    });

    const bulletsAttn = applyCrossTransformerLayers({
        name: modelName + '_bulletsAttn',
        depth: Math.ceil(config.depth / 3),
        heads: config.heads,
        qTok: getAverage(modelName + '_bulletsAvgPool', tokens.bulletsTok),
        kvTok: tokens.bulletsTok,
        kvMask: inputs.bulletsMaskInput,
    });

    const getContextToken = () => {
        return tf.layers.concatenate({name: modelName + '_contextToken' + Math.random(), axis: 1 })
            .apply([
                tokens.tankTok,
                alliesAttn,
                enemiesAttn,
                bulletsAttn,
                // tokens.battleTok,
                // tokens.controllerTok
            ]) as tf.SymbolicTensor;
    }

    const maskLike = new MaskLikeLayer({ name: modelName + '_maskLike' });
    const maskSquash = new MaskSquashLayer({ name: modelName + '_maskSquash' })
    const getContextMask = () => {
        const oneMask = maskLike.apply(tokens.tankTok) as tf.SymbolicTensor;
        const alliesMask = maskSquash.apply(inputs.alliesMaskInput) as tf.SymbolicTensor;
        const enemiesMask = maskSquash.apply(inputs.enemiesMaskInput) as tf.SymbolicTensor;
        const bulletsMask = maskSquash.apply(inputs.bulletsMaskInput) as tf.SymbolicTensor;
        return tf.layers.concatenate({name: modelName + '_contextTokenMask' + Math.random(), axis: 1 })
            .apply([
                oneMask,
                alliesMask,
                enemiesMask,
                bulletsMask,
                // oneMask,
                // oneMask,
            ]) as tf.SymbolicTensor;
    };

    const transformedTanksToken = applySelfTransformLayers(
        modelName + '_transformedTanksToken',
        {
            heads: config.heads,
            depth: config.depth,
            token: getContextToken,
            mask: getContextMask,
            preNorm: true,
        },
    );

    const flattenedFinalToken = tf.layers.flatten({ name: modelName + '_flattenedFinalToken' }).apply(transformedTanksToken) as tf.SymbolicTensor;
    
    const heads = config.headsMLP.map((layers, i) => {
        return applyMLP({
            name: modelName + '_headsMLP_' + i,
            layers,
        }, flattenedFinalToken);
    })

    return {inputs, heads};
}


function getAverage(name: string, token: tf.SymbolicTensor) {
    const average = applyGlobalAverage1d({ name }, token);
    const reshaped = tf.layers.reshape({ targetShape: [1, average.shape[1]!] , name: name + '_reshaped' })
        .apply(average) as tf.SymbolicTensor;
    return reshaped;
}