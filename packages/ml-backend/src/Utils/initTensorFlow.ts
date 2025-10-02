import * as tf from '@tensorflow/tfjs';

tf.enableProdMode();

export async function initTensorFlow() {
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
