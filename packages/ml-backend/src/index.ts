/**
 * ML Backend Entry Point
 * 
 * Node.js backend that:
 * 1. Initializes TensorFlow.js
 * 2. Starts learner agents
 * 3. Receives experience batches via episodeSampleChannel
 * 4. Trains models and publishes weights
 */

// Load environment variables first
import 'dotenv/config';

import { forceExitChannel } from './Common/channels.ts';
import { initTensorFlow } from './Common/initTensorFlow.ts';
import { restoreModels } from './Models/restore.ts';
import { createLearnerManager } from './PPO/Learner/createLearnerManager.ts';
import { createPolicyLearnerAgent } from './PPO/Learner/createPolicyLearnerAgent.ts';
import { createValueLearnerAgent } from './PPO/Learner/createValueLearnerAgent.ts';

console.info('🚀 ML Backend starting...');

// Initialize TensorFlow with Node.js backend
initTensorFlow()
    .then(() => {
        console.info('✅ TensorFlow initialized');

        // Restore models from DB or fallback path
        return restoreModels('./assets/models/v1');
    })
    .then(() => {
        console.info('✅ Models restored');

        // Start learner agents (policy and value)
        createPolicyLearnerAgent();
        createValueLearnerAgent();
        console.info('✅ Learner agents started');

        // Start learner manager (batch processing)
        createLearnerManager();
        console.info('✅ Learner manager started');

        console.info('🎯 Ready to receive experience batches via episodeSampleChannel');
        console.info('   Models will be synced to Supabase automatically');
    })
    .catch((error) => {
        console.error('❌ Failed to initialize:', error);
        process.exit(1);
    });

// Handle force exit signal
forceExitChannel.obs.subscribe(() => {
    console.error('🛑 Force exit signal received');
    process.exit(1);
});

// Handle process termination
process.on('SIGINT', () => {
    console.info('\n🛑 Shutting down gracefully...');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.info('\n🛑 Shutting down gracefully...');
    process.exit(0);
});
