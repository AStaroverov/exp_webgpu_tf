import * as tf from '@tensorflow/tfjs';
import {
    applyCrossTransformerLayers,
    applyMLP, convertInputsToTokens,
    createInputs
} from '../ApplyLayers.ts';

import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { Model } from '../def.ts';
import { VariableLayer } from '../Layers/VariableLayer.ts';
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
    depth: 4,
    headsMLP: [0, 1, 2, 3].map(() =>
        ([
            ['relu', 64],
        ])
    )
};

const valueNetworkConfig: NetworkConfig = {
    dim: 16,
    heads: 1,
    depth: 1,
    headsMLP: [[
        ['relu', 16],
    ]],
};

export function createNetwork(modelName: Model, config: NetworkConfig = modelName === Model.Policy ? policyNetworkConfig : valueNetworkConfig) {
    const depth = config.depth;

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

    const alliesAttnToken = applyCrossTransformerLayers({
        name: modelName + '_alliesAttn',
        heads: config.heads,
        depth,
        qTok: alliesClsToken,
        kvTok: tokens.alliesTok,
        kvMask: inputs.alliesMaskInput,
    });
    const enemiesAttnToken = applyCrossTransformerLayers({
        name: modelName + '_enemiesAttn',
        heads: config.heads,
        depth,
        qTok: enemiesClsToken,
        kvTok: tokens.enemiesTok,
        kvMask: inputs.enemiesMaskInput,
    });
    const bulletsAttnToken = applyCrossTransformerLayers({
        name: modelName + '_bulletsAttn',
        heads: config.heads,
        depth,
        qTok: bulletsClsToken,
        kvTok: tokens.bulletsTok,
        kvMask: inputs.bulletsMaskInput,
    });

    const contextToken = (i: number) => tf.layers.concatenate({name: modelName + '_tankContextToken' + i, axis: 1 }).apply([
        alliesAttnToken,
        enemiesAttnToken,
        bulletsAttnToken,
    ]) as tf.SymbolicTensor;
    const maskSquasher = new MaskSquashLayer({ name: modelName + '_maskSquasher' });
    const contextMask = (i: number) => tf.layers.concatenate({name: modelName + '_tankContextTokenMask' + i, axis: 1 }).apply([
        maskSquasher.apply(inputs.alliesMaskInput) as tf.SymbolicTensor,
        maskSquasher.apply(inputs.enemiesMaskInput) as tf.SymbolicTensor,
        maskSquasher.apply(inputs.bulletsMaskInput) as tf.SymbolicTensor,
    ]) as tf.SymbolicTensor;

   const finalAttn = applyCrossTransformerLayers({
        name: modelName + '_finalAttn',
        heads: config.heads,
        depth,
        qTok: tokens.tankTok,
        kvTok: contextToken,
        kvMask: contextMask,
    });
    
    const heads = config.headsMLP.map((layers, i) => {
        return applyMLP({
            name: modelName + '_headsMLP_' + i,
            layers,
        }, tf.layers.flatten().apply(finalAttn) as tf.SymbolicTensor);
    })

    return { inputs, heads };
}
