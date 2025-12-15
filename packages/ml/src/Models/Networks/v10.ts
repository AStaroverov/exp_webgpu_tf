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
import { ceil } from '../../../../../lib/math.ts';

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
    dim: 16,
    heads: 1,
    depth: 2,
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
                tokens.alliesTok,
                tokens.enemiesTok,
                bulletsToken,
            ]) as tf.SymbolicTensor;
    }

    const maskLike = new MaskLikeLayer({ name: modelName + '_maskLike' });
    const maskSquash = new MaskSquashLayer({ name: modelName + '_maskSquash' });
    const getContextMask = (name: string, i: number) => {
        const oneMask = maskLike.apply(tokens.tankTok) as tf.SymbolicTensor;
        return tf.layers.concatenate({name: name + '_contextTokenMask' + i, axis: 1 })
            .apply([
                oneMask,
                oneMask,
                inputs.alliesMaskInput,
                inputs.enemiesMaskInput,
                maskSquash.apply(inputs.bulletsMaskInput) as tf.SymbolicTensor,
            ]) as tf.SymbolicTensor;
    };

    const decoder = applySelfTransformLayers(
        modelName + '_decoder',
        {
            heads: config.heads,
            depth: ceil(config.depth * 0.5),
            token: getContextToken,
            mask: getContextMask,
            preNorm: true,
        },
    );

    const getStrategyToken = (name: string, noisy: boolean) => {
         const strategy = applySelfTransformLayers(
            name + '_strategyAttn',
            {
                heads: config.heads,
                depth: ceil(config.depth * 0.3),
                token: decoder,
                mask: getContextMask,
                noisy: noisy,
                preNorm: true,
            },
        );
        const token = new SliceLayer({
            name: name + '_strategyToken',
            beginSlice: [0, 0, 0],
            sliceSize: [-1, 1, -1],
        }).apply(strategy) as tf.SymbolicTensor;

        return token;
    }
   
    debugger
    const strategy1 = getStrategyToken(modelName + '_strategy1', false);
    const strategy2 = getStrategyToken(modelName + '_strategy2', true);
    const strategies = tf.layers.concatenate({ name: modelName + '_strategiesConcat', axis: -1 })
        .apply([strategy1, strategy2]) as tf.SymbolicTensor;

    const finalAttn = applySelfTransformLayers(
        modelName + '_finalAttn',
        {
            heads: config.heads,
            depth: ceil(config.depth * 0.2),
            token: strategies,
            preNorm: true,
        },
    );

    const flattenedFinalToken = tf.layers.flatten({ name: modelName + '_flattenedFinalToken' })
        .apply(finalAttn) as tf.SymbolicTensor;

    const len = modelName === Model.Policy ? 4 : 1;
    const heads = Array.from({ length: len }, (_, i) => {
        return applyLaNLayer({
            name: modelName + '_head' + i,
            units: config.dim,
            preNorm: true
        }, flattenedFinalToken);
    });

    return { inputs, heads };
}
