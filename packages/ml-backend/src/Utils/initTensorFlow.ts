import * as tf from '../../../ml-common/tf';

// tf.enableProdMode();

// export async function initTensorFlow() {
//     try {
//         await tf.ready();
//         console.log('TensorFlow.js initialized with Node.js backend');
//         console.log(`TensorFlow.js version: ${tf.version.tfjs}`);
//         console.log(`Backend: ${tf.getBackend()}`);
//         return true;
//     } catch (error) {
//         console.error('Failed to initialize TensorFlow.js Node backend:', error);
//         return false;
//     }
// }


// import * as tf from './tf';

tf.enableProdMode();

export async function initTensorFlow() {
    try {
        await import('@tensorflow/tfjs-backend-webgpu');
        await tf.setBackend('webgpu');
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
