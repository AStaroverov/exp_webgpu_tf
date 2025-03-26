// Main entry point for the Shared Tank PPO system
import { setExperiment } from '../Common/config.ts';
import '@tensorflow/tfjs-backend-wasm';
import { getRLGameManger } from './manager.ts';
import { initTensorFlow } from '../../Common/initTensorFlow.ts';

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