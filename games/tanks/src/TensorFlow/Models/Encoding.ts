import * as tf from '@tensorflow/tfjs';
import { FixedPositionalEncodingLayer } from './FixedPositionalEncodingLayer.ts';
import { RoleEmbeddingLayer } from './RoleEncodingLayer.ts';

export function applyEncoding(token: tf.SymbolicTensor): tf.SymbolicTensor {
    const N = token.shape[1]!;

    const posEmbedding = N === 1
        ? token
        : new FixedPositionalEncodingLayer({
            name: token.name + '_withPos',
        }).apply(token) as tf.SymbolicTensor;

    const roleEmbedding = new RoleEmbeddingLayer({
        name: posEmbedding.name + 'withRole',
    }).apply(posEmbedding) as tf.SymbolicTensor;

    return roleEmbedding;
}
