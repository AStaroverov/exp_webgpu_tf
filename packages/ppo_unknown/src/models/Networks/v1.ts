/**
 * v1 — a simple transformer over the chess-like board, the MVP network for
 * ppo_unknown. Each of the 64 board cells becomes a token (its CHANNELS vector),
 * so the policy can reason about spatial relations via self-attention — closer to
 * tanks' attention nets than a conv stack, but minimal: no masks, no cross-attn.
 *
 *   board [ROWS, COLS, CH] → [ROWS*COLS, CH] tokens → proj dModel → +posenc
 *     → N self-attention layers → mean-pool → feature
 *
 * Contract (consumed by createUnknownNetworks, same as tanks createTankNetworks):
 *   createNetwork(model) → { inputs, heads }
 *     - inputs: Record<string, tf.SymbolicTensor> — Object.values order matches
 *       state/InputTensors.ts.
 *     - heads: tf.SymbolicTensor[] — Policy: one per ACTION_HEAD_DIMS entry;
 *       Value: a single feature (heads[0]).
 */

import * as tf from '@tensorflow/tfjs';
import { Model } from '../../../../ppo/src/models/def.ts';
import {
    applyGlobalAverage1d,
    applySelfTransformLayers,
    createNormalizationLayer,
    tokenProj,
} from '../../../../ppo/src/models/ApplyLayers.ts';
import { FixedPositionalEncodingLayer } from '../../../../ppo/src/models/Layers/FixedPositionalEncodingLayer.ts';
import { ACTION_HEAD_DIMS, BOARD_CELLS, BOARD_CHANNELS } from '../dims.ts';
import { createInputs } from '../Inputs.ts';

type NetworkConfig = { dim: number; heads: number; depth: number };

const policyConfig: NetworkConfig = { dim: 64, heads: 4, depth: 2 };
const valueConfig: NetworkConfig = { dim: 32, heads: 2, depth: 1 };

export function createNetwork(
    modelName: Model,
    config: NetworkConfig = modelName === Model.Policy ? policyConfig : valueConfig,
) {
    const inputs = createInputs(modelName);

    // [B, ROWS, COLS, CH] → [B, ROWS*COLS, CH]: one token per board cell.
    const cellTokens = tf.layers.reshape({
        name: modelName + '_cellTokens',
        targetShape: [BOARD_CELLS, BOARD_CHANNELS],
    }).apply(inputs.boardInput) as tf.SymbolicTensor;

    // Project each cell vector to dModel, then add fixed (sinusoidal) cell-position
    // encoding so attention can tell cells apart.
    let tokens = tokenProj(cellTokens, config.dim, modelName + '_cell');
    tokens = new FixedPositionalEncodingLayer({ name: modelName + '_posEnc' })
        .apply(tokens) as tf.SymbolicTensor;

    const encoded = applySelfTransformLayers(modelName + '_encoder', {
        depth: config.depth,
        heads: config.heads,
        token: tokens,
        preNorm: true,
    });

    const pooled = applyGlobalAverage1d({ name: modelName + '_pool' }, encoded);
    const feature = createNormalizationLayer({ name: modelName + '_featureNorm' })
        .apply(pooled) as tf.SymbolicTensor;

    if (modelName === Model.Policy) {
        const heads = ACTION_HEAD_DIMS.map((_, i) =>                                                         
            tf.layers.dense({                                                                                
                name: modelName + '_headBranch' + i,                                                         
                units: config.dim,                                                                           
                activation: 'relu',                                                                          
            }).apply(feature) as tf.SymbolicTensor,                                                          
        );                 
        return { inputs, heads };
    }

    // Value: single feature; createValueNetwork appends the scalar output.
    return { inputs, heads: [feature] };
}
