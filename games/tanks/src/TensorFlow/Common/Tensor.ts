import * as tf from '@tensorflow/tfjs';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';

function arrayHealthCheck(array: Float32Array | Uint8Array | Int32Array): boolean {
    return array.every(Number.isFinite);
}

export function syncUnwrapTensor<T extends Float32Array | Uint8Array | Int32Array>(tensor: tf.Tensor): T {
    try {
        const value = tensor.dataSync() as T;
        if (!arrayHealthCheck(value)) {
            throw new Error('Invalid tensor value');
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
            throw new Error('Invalid tensor value');
        }
        return value;
    } finally {
        tensor.dispose();
    }
}

export function onReadyRead() {
    if (tf.getBackend() === 'webgpu') {
        return (tf.backend() as WebGPUBackend).device.queue.onSubmittedWorkDone();
    }
    return Promise.resolve();
}
