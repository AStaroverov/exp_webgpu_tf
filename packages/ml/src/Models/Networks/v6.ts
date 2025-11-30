import * as tf from '@tensorflow/tfjs';
import {
    applyLaNLayer,
    applySelfTransformLayers,
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
    lanUnits: number;
};

const policyNetworkConfig: NetworkConfig = {
    dim: 32,
    heads: 4,
    depth: 6,
    lanUnits: 32,
};

const valueNetworkConfig: NetworkConfig = {
    dim: 16,
    heads: 1,
    depth: 1,
    lanUnits: 16,
};

export function createNetwork(modelName: Model, config: NetworkConfig = modelName === Model.Policy ? policyNetworkConfig : valueNetworkConfig) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, config.dim);

    // CLS Token
    const clsToken = new VariableLayer({
        name: modelName + '_clsToken',
        shape: [1, config.dim],
        initializer: 'truncatedNormal'
    }).apply(tokens.tankTok) as tf.SymbolicTensor;

    // Combine all tokens: CLS, Tank, Allies, Enemies, Bullets
    const allTokens = tf.layers.concatenate({name: modelName + '_allTokens', axis: 1 })
        .apply([
            clsToken,
            tokens.tankTok,
            tokens.alliesTok,
            tokens.enemiesTok,
            tokens.bulletsTok
        ]) as tf.SymbolicTensor;

    // Create masks
    const maskLike = new MaskLikeLayer({ name: modelName + '_maskLike' });
    const clsMask = maskLike.apply(clsToken) as tf.SymbolicTensor;
    const tankMask = maskLike.apply(tokens.tankTok) as tf.SymbolicTensor;
    
    const getAllMasks = (i: number) => tf.layers.concatenate({name: modelName + '_allMasks' + i, axis: 1 })
        .apply([
            clsMask,
            tankMask,
            inputs.alliesMaskInput,
            inputs.enemiesMaskInput,
            inputs.bulletsMaskInput
        ]) as tf.SymbolicTensor;

    // Transformer
    const transformedTokens = applySelfTransformLayers(
        modelName + '_transformedTokens',
        {
            heads: config.heads,
            depth: config.depth,
            token: allTokens,
            mask: getAllMasks,
            preNorm: true,
        },
    );

    // Extract CLS token (first token)
    const transformedClsToken = new SliceLayer({
        name: modelName + '_transformedClsToken',
        beginSlice: [0, 0, 0],
        sliceSize: [-1, 1, -1],
    }).apply(transformedTokens) as tf.SymbolicTensor;

    const flattenedCls = tf.layers.flatten({ name: modelName + '_flattenedCls' })
        .apply(transformedClsToken) as tf.SymbolicTensor;

    // Heads
    const numHeads = modelName === Model.Policy ? 4 : 1;
    const heads = [];
    
    for (let i = 0; i < numHeads; i++) {
        const head = applyLaNLayer({
            name: `${modelName}_head${i}`,
            units: config.lanUnits,
            preNorm: true
        }, flattenedCls);
        heads.push(head);
    }

    return { inputs, heads };
}
