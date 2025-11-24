import * as tf from '@tensorflow/tfjs';
import {
    applyCrossTransformerLayer,
    applyMLP, convertInputsToTokens,
    createInputs
} from '../ApplyLayers.ts';

import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { Model } from '../def.ts';
import { VariableLayer } from '../Layers/VariableLayer.ts';

type NetworkConfig = {
    dim: number;
    heads: number;
    headsMLP: ([ActivationIdentifier, number][])[];
};

type policyNetworkConfig = NetworkConfig

const policyNetworkConfig: policyNetworkConfig = {
    dim: 16,
    heads: 1,
    headsMLP: [0, 1, 2, 3].map(() =>
        ([
            ['relu', 64],
        ])
    )
};

const valueNetworkConfig: NetworkConfig = {
    dim: 16,
    heads: 1,
    headsMLP: [[
        ['relu', 16],
    ]],
};

export function createNetwork(modelName: Model, config: NetworkConfig = modelName === Model.Policy ? policyNetworkConfig : valueNetworkConfig) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, config.dim);

    const alliesClsToken = new VariableLayer({
        name: modelName + '_alliesClsToken',
        shape: [1, config.dim],
        initializer: 'truncatedNormal'
    }).apply(tokens.tankTok) as tf.SymbolicTensor;
    const enemiesClsToken = new VariableLayer({
        name: modelName + '_enemiesClsToken',
        shape: [1, config.dim],
        initializer: 'truncatedNormal'
    }).apply(tokens.tankTok) as tf.SymbolicTensor;
    const bulletsClsToken = new VariableLayer({
        name: modelName + '_bulletsClsToken',
        shape: [1, config.dim],
        initializer: 'truncatedNormal'
    }).apply(tokens.tankTok) as tf.SymbolicTensor;

    const alliesAttnToken = applyCrossTransformerLayer({
        name: modelName + '_alliesAttn',
        heads: config.heads,
        qTok: alliesClsToken,
        kvTok: tokens.alliesTok,
        kvMask: inputs.alliesMaskInput,
    });
    const enemiesAttnToken = applyCrossTransformerLayer({
        name: modelName + '_enemiesAttn',
        heads: config.heads,
        qTok: enemiesClsToken,
        kvTok: tokens.enemiesTok,
        kvMask: inputs.enemiesMaskInput,
    });
    const bulletsAttnToken = applyCrossTransformerLayer({
        name: modelName + '_bulletsAttn',
        heads: 1,
        qTok: bulletsClsToken,
        kvTok: tokens.bulletsTok,
        kvMask: inputs.bulletsMaskInput,
    });

    const contextToken = tf.layers.concatenate({name: modelName + '_tankContextToken', axis: 1 }).apply([
        tokens.tankTok,
        alliesAttnToken,
        enemiesAttnToken,
        bulletsAttnToken,
    ]) as tf.SymbolicTensor;

    // Apply LTSM layer
    const lstm = tf.layers.lstm({
        name: modelName + '_lstmLayer',
        units: config.dim * 4,
        returnState: false,
        returnSequences: false,
    }).apply(contextToken) as tf.SymbolicTensor;
    
    const heads = config.headsMLP.map((layers, i) => {
        return applyMLP({
            name: modelName + '_headsMLP_' + i,
            layers,
        }, lstm);
    })

    return { inputs, heads };
}
