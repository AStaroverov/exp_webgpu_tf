import * as tf from '@tensorflow/tfjs';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import { MAX_STEPS } from './consts.ts';
import { BATCH_SIZE, PPOAgent } from './PPOAgent.ts';
import { runEpisode } from './runEpisode.ts';

setWasmPaths('/node_modules/@tensorflow/tfjs-backend-wasm/dist/');
await tf.setBackend('wasm');

// Main enhanced PPO training function with better tracking and checkpointing
async function trainPPO(checkpointInterval: number = 5): Promise<void> {
    console.log('Starting enhanced PPO training...');

    const agent = new PPOAgent();
    let episodesCompleted = 0;
    let totalReward = 0;
    let bestEpisodeReward = -Infinity;

    // Track metrics for visualization and analysis
    const trainingMetrics = {
        episodeRewards: [] as number[],
        actorLosses: [] as number[],
        criticLosses: [] as number[],
        avgRewards: [] as number[],
        episodeLengths: [] as number[],
    };

    try {
        // Try to load existing models
        const loaded = await agent.loadModels();
        if (loaded) {
            console.log('Loaded existing models, continuing training from episode', agent.episodeCount);
            episodesCompleted = agent.episodeCount;
        } else {
            console.log('Starting with new models');
        }

        // Training loop with improved checkpointing and error handling
        for (let i = episodesCompleted; i < Infinity; i++) {
            console.log(`Starting episode ${ i + 1 }`);

            // Track episode start time for performance monitoring
            const episodeStartTime = performance.now();

            try {
                // Run episode with a reasonable step limit
                const episodeReward = await runEpisode(agent, MAX_STEPS);
                totalReward += episodeReward;

                // Track episode duration for metrics
                const episodeDuration = performance.now() - episodeStartTime;
                console.log(`Episode ${ i + 1 } completed in ${ (episodeDuration / 1000).toFixed(2) }s with reward: ${ episodeReward.toFixed(2) }`);

                // Update metrics
                trainingMetrics.episodeRewards.push(episodeReward);
                trainingMetrics.episodeLengths.push(MAX_STEPS); // This will be actual steps if terminated early
                trainingMetrics.avgRewards.push(totalReward / (i + 1 - episodesCompleted));

                // Train after each episode
                const trainingStartTime = performance.now();

                // Track losses for this training session
                let actorLossSum = 0;
                let criticLossSum = 0;
                let trainingIterations = 0;

                // Multiple training iterations if enough data is available
                const minBatchesForTraining = 3; // Train only if we have at least 3x batch size data
                const minSamplesRequired = BATCH_SIZE * minBatchesForTraining;

                if (agent.buffer.size >= minSamplesRequired) {
                    // Number of training iterations based on buffer size
                    const trainIterations = Math.min(
                        Math.floor(agent.buffer.size / BATCH_SIZE),
                        5, // Cap maximum iterations per episode
                    );

                    for (let iter = 0; iter < trainIterations; iter++) {
                        // Train and get average losses
                        const { actorLoss, criticLoss } = await agent.train();

                        if (actorLoss !== undefined && criticLoss !== undefined) {
                            actorLossSum += actorLoss;
                            criticLossSum += criticLoss;
                            trainingIterations++;
                        }
                    }

                    // Log average losses
                    if (trainingIterations > 0) {
                        const avgActorLoss = actorLossSum / trainingIterations;
                        const avgCriticLoss = criticLossSum / trainingIterations;

                        trainingMetrics.actorLosses.push(avgActorLoss);
                        trainingMetrics.criticLosses.push(avgCriticLoss);

                        console.log(`Training complete (${ trainingIterations } iterations, ${ (performance.now() - trainingStartTime).toFixed(0) }ms)`);
                        console.log(`Average Actor Loss: ${ avgActorLoss.toFixed(4) }, Critic Loss: ${ avgCriticLoss.toFixed(4) }`);
                    } else {
                        console.log('No valid training occurred this episode');
                    }
                } else {
                    console.log(`Not enough samples for training: ${ agent.buffer.size }/${ minSamplesRequired }`);
                }

                // Update episode counter
                agent.episodeCount = i + 1;

                // Check if this is the best episode so far
                if (episodeReward > bestEpisodeReward) {
                    bestEpisodeReward = episodeReward;

                    // Save best model separately
                    await agent.saveModels('best');
                    console.log(`New best model saved with reward: ${ bestEpisodeReward.toFixed(2) }`);
                }

                // Regular checkpointing
                if ((i + 1) % checkpointInterval === 0) {
                    await agent.saveModels();
                    console.log(`Checkpoint saved at episode ${ i + 1 }`);

                    // Save metrics
                    try {
                        localStorage.setItem('tank-training-metrics', JSON.stringify(trainingMetrics));
                    } catch (error) {
                        console.warn('Failed to save metrics:', error);
                    }

                    if ((i + 1) % (checkpointInterval * 20) === 0) {
                        // because of tensoflow.js memory leak
                        window.location.reload();
                    }
                }
            } catch (error) {
                console.error(`Error in episode ${ i + 1 }:`, error);

                // Try to save current state before potential recovery
                try {
                    await agent.saveModels('recovery');
                    console.log('Recovery checkpoint saved');
                } catch (saveError) {
                    console.error('Failed to save recovery checkpoint:', saveError);
                }

                // Wait a moment before continuing to next episode
                await new Promise(resolve => setTimeout(resolve, 5000));
                // Skip to next episode instead of crashing
            }
        }

        // Final save and cleanup
        await agent.saveModels();

        // Save final metrics
        try {
            localStorage.setItem('tank-training-metrics', JSON.stringify(trainingMetrics));
        } catch (error) {
            console.warn('Failed to save final metrics:', error);
        }

        console.log('Training completed successfully!');
        console.log(`Total episodes: ${ agent.episodeCount }`);
        console.log(`Best episode reward: ${ bestEpisodeReward.toFixed(2) }`);
    } catch (error) {
        console.error('Critical error during training:', error);

        // Try emergency save
        try {
            await agent.saveModels('emergency');
            console.log('Emergency save completed');
        } catch (saveError) {
            console.error('Failed to perform emergency save:', saveError);
        }

        // Wait before potentially reloading page
        console.log('Reloading page in 10 seconds...');
        setTimeout(() => {
            window.location.reload();
        }, 10_000);
    }
}

trainPPO(5);