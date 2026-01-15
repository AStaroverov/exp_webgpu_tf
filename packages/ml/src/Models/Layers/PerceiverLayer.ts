import * as tf from '@tensorflow/tfjs';
import { MaskLikeLayer } from "./MaskLikeLayer";
import { applyCrossAttentionLayer, applySelfTransformerLayer } from "../ApplyLayers";

export function applyPerceiverLayer({
    name,
    depth,
    heads,
    qTok,
    kvTok,
    qMask,
    kvMask,
    preNorm = false,
}: {
    name: string, 
    depth: number,
    heads: number,
    qTok: tf.SymbolicTensor | ((name: string, i: number) => tf.SymbolicTensor),
    kvTok: tf.SymbolicTensor | ((name: string, i: number) => tf.SymbolicTensor),
    qMask?: tf.SymbolicTensor | ((name: string, i: number) => tf.SymbolicTensor),
    kvMask?: tf.SymbolicTensor | ((name: string, i: number) => tf.SymbolicTensor),
    preNorm?: boolean,
}) {
    let x = typeof qTok === 'function' ? qTok(name, 0) : qTok;

    for (let i = 0; i < depth; i++) {
        const kvTokI = typeof kvTok === 'function' ? kvTok(name, i) : kvTok;
        const qMaskI = typeof qMask === 'function'
            ? qMask(name, i)
            : qMask ?? new MaskLikeLayer({ name: name + '_qMaskLike' + i }).apply(x) as tf.SymbolicTensor;
        const kvMaskI = typeof kvMask === 'function'
            ? kvMask(name, i)
            : kvMask ?? new MaskLikeLayer({ name: name + '_kvMaskLike' + i }).apply(kvTokI) as tf.SymbolicTensor;
        
        x = applyCrossAttentionLayer({
            name: `${name}/cross/depth${i}`,
            heads,
            qTok: x,
            kvTok: kvTokI,
            qMask: qMaskI,
            kvMask: kvMaskI,
            preNorm,
        });
        x = applySelfTransformerLayer({
            name: `${name}/self/depth${i}`,
            heads,
            token: x,
            mask: qMaskI,
            preNorm,
        });
    }

    return x;
}