import * as tf from '@tensorflow/tfjs';
import {
    applyCrossAttentionLayer, applyLaNLayer, applySelfTransformLayers,
    convertInputsToTokens, createInputs
} from '../ApplyLayers.ts';
import { MaskLikeLayer } from '../Layers/MaskLikeLayer.ts';

import { Model } from '../def.ts';
import { VariableLayer } from '../Layers/VariableLayer.ts';
import { SliceLayer } from '../Layers/SliceLayer.ts';
import { MaskSquashLayer } from '../Layers/MaskSquashLayer.ts';

type NetworkConfig = {
    dim: number;
    heads: number;
    depth: number;
};

type policyNetworkConfig = NetworkConfig

const policyNetworkConfig: policyNetworkConfig = {
    dim: 16,
    heads: 1,
    depth: 12,
};

const valueNetworkConfig: NetworkConfig = {
    dim: 8,
    heads: 1,
    depth: 3,
};

export function createNetwork(modelName: Model, config: NetworkConfig = modelName === Model.Policy ? policyNetworkConfig : valueNetworkConfig) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, config.dim);

    const clsBulletsToken = new VariableLayer({
        name: modelName + '_clsBulletsToken',
        shape: [1, config.dim],
        initializer: 'truncatedNormal'
    }).apply(tokens.bulletsTok) as tf.SymbolicTensor;
    const bulletsToken = applyCrossAttentionLayer({
        name: modelName + '_clsBulletsAttention',
        heads: config.heads,
        qTok: clsBulletsToken,
        kvTok: tokens.bulletsTok,
        kvMask: inputs.bulletsMaskInput,
    });

    const clsToken = new VariableLayer({
        name: modelName + '_clsToken',
        shape: [1, config.dim],
        initializer: 'truncatedNormal'
    }).apply(tokens.tankTok) as tf.SymbolicTensor;
    
    const getContextToken = (name: string, i: number) => {
        return tf.layers.concatenate({name: name + '_contextToken' + i, axis: 1 })
            .apply([
                clsToken,
                tokens.tankTok,
                tokens.turretRaysTok,
                bulletsToken,
                tokens.envRaysTok,
                tokens.alliesTok,
                tokens.enemiesTok,
            ]) as tf.SymbolicTensor;
    }

    const maskLike = new MaskLikeLayer({ name: modelName + '_maskLike' });
    const maskSquash = new MaskSquashLayer({ name: modelName + '_maskSquash' });
    const getContextMask = (name: string, i: number) => {
        const oneMask = maskLike.apply(tokens.tankTok) as tf.SymbolicTensor;
        const bulletsMask = maskSquash.apply(inputs.bulletsMaskInput) as tf.SymbolicTensor;
        const envRaysMask = maskLike.apply(tokens.envRaysTok) as tf.SymbolicTensor;
        const turretRaysMask = maskLike.apply(tokens.turretRaysTok) as tf.SymbolicTensor;
        
        return tf.layers.concatenate({name: name + '_contextTokenMask' + i, axis: 1 })
            .apply([
                oneMask,
                oneMask,
                turretRaysMask,
                bulletsMask,
                envRaysMask,
                inputs.alliesMaskInput,
                inputs.enemiesMaskInput,
            ]) as tf.SymbolicTensor;
    };

    const transformer = applySelfTransformLayers(
        modelName + '_inputTransformer',
        {
            heads: config.heads,
            depth: config.depth,
            token: getContextToken,
            mask: getContextMask,
            preNorm: true,
        },
    );

    const finalToken = new SliceLayer({
        name: modelName + '_finalToken',
        beginSlice: [0, 0, 0],
        sliceSize: [-1, 2, -1],
    }).apply(transformer) as tf.SymbolicTensor;

    const flattenedFinalToken = tf.layers.flatten({ name: modelName + '_flattenedFinalToken' })
        .apply(finalToken) as tf.SymbolicTensor;

    const len = modelName === Model.Policy ? 4 : 1;
    const heads = Array.from({ length: len }, (_, i) => {
        return applyLaNLayer({
            name: modelName + '_head' + i,
            units: config.dim * 2,
            preNorm: true
        }, flattenedFinalToken);
    });

    return { inputs, heads };
}
