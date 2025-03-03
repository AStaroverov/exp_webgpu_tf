// Main entry point for the Shared Tank RL system
import { EXPERIMENTS, setExperiment } from './experiment-config';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { SharedRLGameManager } from './SharedRLGameManager.ts';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';

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
async function initSharedRLSystem(options: {
    isTraining?: boolean;
    experimentName?: string;
} = {}) {
    const {
        isTraining = true,
        experimentName = 'default',
    } = options;

    console.log('===================================');
    console.log('Initializing Shared Tank RL System');
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
    console.log('Using experiment configuration: ' + experimentName);
    // console.log(getExperimentSettings());

    // Start the game
    try {
        const gameManager = new SharedRLGameManager(isTraining);
        await gameManager.init();
        gameManager.start();

        console.log('Shared Tank RL System successfully initialized');

        return gameManager;
    } catch (error) {
        console.error('Failed to start Shared Tank RL Game:', error);
        return null;
    }
}

// Export available experiments for UI selection
export const availableExperiments = Object.keys(EXPERIMENTS).map(key => ({
    name: key,
    description: EXPERIMENTS[key].description,
}));

// Export the main initialization function
export { initSharedRLSystem };
