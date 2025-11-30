import * as tf from '@tensorflow/tfjs';
import {
    applyCrossAttentionLayer,
    applyLaNLayer, applySelfTransformLayers,
    convertInputsToTokens,
    createInputs
} from '../ApplyLayers.ts';
import { MaskLikeLayer } from '../Layers/MaskLikeLayer.ts';

import { Model } from '../def.ts';
import { VariableLayer } from '../Layers/VariableLayer.ts';
import { SliceLayer } from '../Layers/SliceLayer.ts';

type NetworkConfig = {
    dim: number;
    heads: number;
    depth: number;
};

type policyNetworkConfig = NetworkConfig

const policyNetworkConfig: policyNetworkConfig = {
    dim: 32,
    heads: 4,
    depth: 8,
};

const valueNetworkConfig: NetworkConfig = {
    dim: 16,
    heads: 1,
    depth: 2,
};

export function createNetwork(modelName: Model, config: NetworkConfig = modelName === Model.Policy ? policyNetworkConfig : valueNetworkConfig) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, config.dim);

    const clsToken = new VariableLayer({
        name: modelName + '_clsToken',
        shape: [1, config.dim],
        initializer: 'truncatedNormal'
    }).apply(tokens.tankTok) as tf.SymbolicTensor;
    
    const getContextToken = (i: number) => {
        return tf.layers.concatenate({name: modelName + '_contextToken' + i, axis: 1 })
            .apply([
                clsToken,
                tokens.tankTok,
                tokens.alliesTok,
                tokens.enemiesTok,
            ]) as tf.SymbolicTensor;
    }

    const maskLike = new MaskLikeLayer({ name: modelName + '_maskLike' });
    const oneMask = maskLike.apply(tokens.tankTok) as tf.SymbolicTensor;
    const getContextMask = (i: number) => {
        return tf.layers.concatenate({name: modelName + '_contextTokenMask' + i, axis: 1 })
            .apply([
                oneMask,
                oneMask,
                inputs.alliesMaskInput,
                inputs.enemiesMaskInput,
            ]) as tf.SymbolicTensor;
    };

    const transformedTokens = applySelfTransformLayers(
        modelName + '_transformedTokens',
        {
            heads: config.heads,
            depth: config.depth,
            token: getContextToken,
            mask: getContextMask,
            preNorm: true,
        },
    );

    const transformedClsToken = new SliceLayer({
        name: modelName + '_transformedClsToken',
        beginSlice: [0, 0, 0],
        sliceSize: [-1, 1, -1],
    }).apply(transformedTokens) as tf.SymbolicTensor;

    const flattenedTransformedClsToken = tf.layers.flatten({ name: modelName + '_flattenedTransformedClsToken' })
        .apply(transformedClsToken) as tf.SymbolicTensor;

    const transformedTankToken = new SliceLayer({
        name: modelName + '_transformedTankToken',
        beginSlice: [0, 1, 0],
        sliceSize: [-1, 1, -1],
    }).apply(transformedTokens) as tf.SymbolicTensor;

    const attentionToBullets = applyCrossAttentionLayer({
        name: modelName + '_attentionToBullets',
        heads: config.heads,
        qTok: transformedTankToken,
        kvTok: tokens.bulletsTok,
        kvMask: inputs.bulletsMaskInput,
    });

    const flattenedAttentionToBullets = tf.layers.flatten({ name: modelName + '_flattenedAttentionToBullets' })
        .apply(attentionToBullets) as tf.SymbolicTensor;

    const getSingleToken = (name: string) => {
        const concatenated = tf.layers.concatenate({name: name + '_concatenated', axis: 1 })
            .apply([flattenedTransformedClsToken, flattenedAttentionToBullets]) as tf.SymbolicTensor;
        return concatenated;
    }

    const getHeadToken = (i: number) => {
        if (i === 0 || i === 1) {
            return flattenedAttentionToBullets;
        }
        if (i === 2 || i === 3) {
            return flattenedTransformedClsToken;
        }
        throw new Error('Invalid head index');
    }

    const heads = Array.from({length: modelName === Model.Policy ? 4 : 1 }, (_, i) => {
        const token = modelName === Model.Policy ? getHeadToken(i) : getSingleToken(modelName + '_headToken_' + i);
        return applyLaNLayer({
            name: `${modelName}_head${i}`,
            units: config.dim,
            preNorm: true
        }, token);
    });

    return { inputs, heads };
}
