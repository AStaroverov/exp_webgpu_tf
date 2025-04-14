import * as tf from '@tensorflow/tfjs';
import { arrayHealthCheck } from '../PPO/train.ts';

export function syncUnwrapTensor<T extends Float32Array | Uint8Array | Int32Array>(tensor: tf.Tensor): T {
    try {
        const value = tensor.dataSync() as T;
        if (!arrayHealthCheck(value)) {
            throw new Error('Invalid loss value');
        }
        return value;
    } finally {
        tensor.dispose();
    }
}

export async function asyncUnwrapTensor<T extends Float32Array | Uint8Array | Int32Array>(tensor: tf.Tensor): Promise<T> {
    try {
        const value = await tensor.data() as T;
        if (!arrayHealthCheck(value)) {
            throw new Error('Invalid loss value');
        }
        return value;
    } finally {
        tensor.dispose();
    }
}

