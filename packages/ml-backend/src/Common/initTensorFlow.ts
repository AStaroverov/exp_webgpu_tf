import * as tf from '@tensorflow/tfjs';
import { isNode } from '../../../../lib/detect.ts';

tf.enableProdMode();

export async function initTensorFlow(type: 'wasm' | 'webgpu' | 'node' = 'wasm') {
    if (isNode) {
        // In Node.js, use tfjs-node backend
        try {
            await import('@tensorflow/tfjs-node');
            await tf.ready();
            console.log('TensorFlow.js initialized with Node.js backend');
            console.log(`TensorFlow.js version: ${tf.version.tfjs}`);
            console.log(`Backend: ${tf.getBackend()}`);
            return true;
        } catch (error) {
            console.error('Failed to initialize TensorFlow.js Node backend:', error);
            return false;
        }
    }

    // Browser initialization
    if (tf.getBackend() === type) return;

    try {
        if (type === 'wasm') {
            const { setWasmPath } = await import('@tensorflow/tfjs-backend-wasm');
            setWasmPath('/assets/wasm/tfjs-backend-wasm-simd.wasm');
            await tf.setBackend('wasm');
        }
        if (type === 'webgpu') {
            await import('@tensorflow/tfjs-backend-webgpu');
            await tf.setBackend('webgpu');
        }
        await tf.ready();
        console.log('TensorFlow.js initialized with WASM backend');

        // Log version info for debugging
        console.log(`TensorFlow.js version: ${tf.version.tfjs}`);
        console.log(`Backend: ${tf.getBackend()}`);

        return true;
    } catch (error) {
        console.error('Failed to initialize TensorFlow.js:', error);
        return false;
    }
}
