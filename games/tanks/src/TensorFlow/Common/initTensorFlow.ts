import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import * as tf from '@tensorflow/tfjs';

export async function initTensorFlow() {
    try {
        // Configure TensorFlow.js to use WASM backend
        setWasmPaths('/node_modules/@tensorflow/tfjs-backend-wasm/dist/');
        await tf.setBackend('wasm');
        await tf.ready();
        console.log('TensorFlow.js initialized with WASM backend');

        // Log version info for debugging
        console.log(`TensorFlow.js version: ${ tf.version.tfjs }`);
        console.log(`Backend: ${ tf.getBackend() }`);

        return true;
    } catch (error) {
        console.error('Failed to initialize TensorFlow.js:', error);
        return false;
    }
}
