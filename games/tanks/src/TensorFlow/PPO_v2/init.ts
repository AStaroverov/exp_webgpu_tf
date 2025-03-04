// Main entry point for the Shared Tank PPO system
import { setExperiment } from './experiment-config';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import { getRLGameManger } from './manager.ts';

// Initialize TensorFlow.js with WASM backend
async function initTensorFlow() {
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

// Main initialization function
async function initSharedPPOSystem(options: {
    isTraining?: boolean;
    experimentName?: string;
} = {}) {
    const {
        isTraining = true,
        experimentName = 'default',
    } = options;

    console.log('===================================');
    console.log('Initializing Shared Tank PPO System');
    console.log(`Mode: ${ isTraining ? 'Training' : 'Evaluation' }`);
    console.log(`Experiment: ${ experimentName }`);
    console.log('===================================');

    // Initialize TensorFlow.js
    const tfInitialized = await initTensorFlow();
    if (!tfInitialized) {
        console.error('Failed to initialize TensorFlow.js, aborting');
        return null;
    }

    // Set experiment configuration
    setExperiment(experimentName);
    console.log('Using PPO experiment configuration: ' + experimentName);

    // Start the game
    try {
        const gameManager = getRLGameManger();
        await gameManager.init();
        gameManager.start();

        console.log('Shared Tank PPO System successfully initialized');

        return gameManager;
    } catch (error) {
        console.error('Failed to start Shared Tank PPO Game:', error);
        return null;
    }
}

// Export the main initialization function
export { initSharedPPOSystem };