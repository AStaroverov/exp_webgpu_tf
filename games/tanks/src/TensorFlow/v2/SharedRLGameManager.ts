import * as tf from '@tensorflow/tfjs';
import { getCurrentExperiment, getExperimentSettings, setExperiment } from './experiment-config';
import { entityExists } from 'bitecs';
import {
    cleanupAllRL,
    deactivateTank,
    getActiveTankCount,
    initSharedRLController,
    isActiveTank,
    logEpisodeCompletion,
    registerTank,
    saveSharedModel,
    trainSharedModel,
    updateTankWithSharedRL,
} from './controller.ts';
import { createBattlefield } from '../Common/createBattlefield.ts';
import { MAX_STEPS, TANK_COUNT_SIMULATION, TICK_TIME_REAL, TICK_TIME_SIMULATION } from '../Common/consts.ts';
import { getSharedAgent } from './agent.ts';
import { macroTasks } from '../../../../../lib/TasksScheduler/macroTasks.ts';
import { RingBuffer } from 'ring-buffer-ts';


// Main class to manage reinforcement learning integration with the game
export class SharedRLGameManager {
    private battlefield!: ReturnType<typeof createBattlefield>;
    private frameCount: number = 0;
    private episodeCount: number = 0;
    private isTraining: boolean = true;
    private gameLoopRunning: boolean = false;
    private episodeStats = new RingBuffer<{
        episodeNumber: number;
        duration: number;
        avgReward: number;
        survivingTanks: number;
        epsilon: number;
        loss: number;
    }>(10);
    private stopFrameInterval: VoidFunction | null = null;
    private stopTrainingInterval: VoidFunction | null = null;

    // Performance tracking
    private lastFrameTime: number = 0;
    private frameTimeSamples: number[] = [];
    private trainingTimeSamples: number[] = [];

    constructor(isTraining: boolean = true) {
        this.isTraining = isTraining;
        console.log(`SharedRLGameManager initialized in ${ isTraining ? 'training' : 'evaluation' } mode`);
        console.log(`Using experiment: ${ getCurrentExperiment().name }`);
    }

    // Initialize the game environment
    async init() {
        console.log('Initializing shared tank RL game environment...');

        // Ensure TensorFlow.js is ready
        await tf.ready();
        console.log('TensorFlow.js ready');

        // Initialize shared RL controller
        await initSharedRLController(!this.isTraining);

        // Initialize battlefield with tanks
        this.resetEnvironment();

        return this;
    }

    // Reset environment for a new episode
    resetEnvironment() {
        // Create new battlefield
        this.battlefield?.destroy();
        this.battlefield = createBattlefield(TANK_COUNT_SIMULATION);

        // Register each tank with the RL system
        for (const tankEid of this.battlefield.tanks) {
            registerTank(tankEid);
        }

        // Reset frame counter
        this.frameCount = 0;
        this.lastFrameTime = performance.now();
        this.frameTimeSamples = [];

        // Increment episode counter
        this.episodeCount++;

        console.log(`Environment reset for episode ${ this.episodeCount } with ${ this.battlefield.tanks.length } tanks`);

        return this.battlefield;
    }

    // Start the game loop
    start() {
        if (this.gameLoopRunning) {
            console.warn('Game loop already running');
            return;
        }

        console.log('Starting shared tank RL game loop...');
        this.gameLoopRunning = true;
        this.gameLoop();

        // Start training loop if in training mode
        if (this.isTraining) {
            this.startTrainingLoop();
        }

        return this;
    }

    // Stop the game loop
    stop() {
        this.gameLoopRunning = false;
        if (this.stopFrameInterval !== null) {
            this.stopFrameInterval();
            this.stopFrameInterval = null;
        }

        // Stop training loop
        if (this.stopTrainingInterval !== null) {
            this.stopTrainingInterval();
            this.stopTrainingInterval = null;
        }

        console.log('Shared tank RL game loop stopped');

        return this;
    }

    // Get current statistics
    getStats() {
        const agent = getSharedAgent();
        const stats = agent.getStats();

        // Calculate average frame time
        const avgFrameTime = this.frameTimeSamples.length > 0
            ? this.frameTimeSamples.reduce((a, b) => a + b, 0) / Math.max(1, this.frameTimeSamples.length)
            : 0;

        return {
            ...stats,
            episodeCount: this.episodeCount,
            frameCount: this.frameCount,
            activeTanks: getActiveTankCount(),
            avgFrameTime,
            experiment: getCurrentExperiment().name,
        };
    }

    // Save models
    async saveModels() {
        try {
            const saved = await saveSharedModel();
            console.log(`Shared model saved: ${ saved }`);
            return saved;
        } catch (error) {
            console.error('Error saving shared model:', error);
            return false;
        }
    }

    // Clean up resources
    cleanup() {
        this.stop();
        cleanupAllRL();
        console.log('Shared tank RL game manager cleaned up');
    }

    // Get the current battlefield
    getBattlefield() {
        return this.battlefield;
    }

    // Pause the simulation
    pause() {
        if (!this.gameLoopRunning) return;

        this.stop();
        console.log('Simulation paused');
    }

    // Resume the simulation
    resume() {
        if (this.gameLoopRunning) return;

        this.gameLoopRunning = true;
        this.gameLoop();

        if (this.isTraining) {
            this.startTrainingLoop();
        }

        console.log('Simulation resumed');
    }

    // Switch experiment during runtime
    switchExperiment(experimentName: string) {
        // Set the new experiment
        setExperiment(experimentName);
        console.log(`Switched to experiment: ${ experimentName }`);

        // Log current settings
        console.log(getExperimentSettings());

        return getCurrentExperiment();
    }

    // Get episode history
    getEpisodeHistory() {
        return this.episodeStats.toArray();
    }

    // Export training data for visualization or analysis
    exportTrainingData() {
        return {
            episodes: this.episodeStats.toArray(),
            performance: {
                frameTimes: this.frameTimeSamples,
                trainingTimes: this.trainingTimeSamples,
            },
            experiment: getCurrentExperiment(),
        };
    }

    // Start background training loop
    private startTrainingLoop() {
        if (this.stopTrainingInterval !== null) {
            this.stopTrainingInterval();
            this.stopTrainingInterval = null;
        }

        console.log('Starting background training loop');

        // Train every 100ms to avoid blocking the main thread
        this.stopTrainingInterval = macroTasks.addInterval(async () => {
            if (!this.gameLoopRunning) return;

            const startTime = performance.now();

            // Train the shared model
            const loss = await trainSharedModel();

            const endTime = performance.now();
            const trainingTime = endTime - startTime;

            // Track training time
            this.trainingTimeSamples.push(trainingTime);
            if (this.trainingTimeSamples.length > 100) {
                this.trainingTimeSamples.shift();
            }

            // Log training performance occasionally
            if (this.frameCount % 100 === 0) {
                const avgTrainingTime = this.trainingTimeSamples.reduce((a, b) => a + b, 0) /
                    Math.max(1, this.trainingTimeSamples.length);
                console.log(`Avg training time: ${ avgTrainingTime.toFixed(2) }ms, Loss: ${ loss }`);
            }
        }, 100);
    }

    // Main game loop
    private gameLoop() {
        this.stopFrameInterval = macroTasks.addInterval(() => {
            if (!this.gameLoopRunning) {
                this.stopFrameInterval?.();
                this.stopFrameInterval = null;
                return;
            }

            const now = performance.now();
            const frameDelta = now - this.lastFrameTime;
            this.lastFrameTime = now;

            // Track frame time for performance monitoring
            if (this.frameCount > 10) { // Skip first few frames
                this.frameTimeSamples.push(frameDelta);
                if (this.frameTimeSamples.length > 100) {
                    this.frameTimeSamples.shift(); // Keep only the last 100 samples
                }
            }

            // Update frame counter
            this.frameCount++;

            if (this.frameCount % 60 === 0) {
                // Get active tanks (with health > 0)
                const activeTanks = this.battlefield.tanks.filter(tankEid => {
                    if (entityExists(this.battlefield.world, tankEid)) return true;
                    if (isActiveTank(tankEid)) deactivateTank(tankEid);
                    return false;
                });

                // Update each tank's RL controller
                const width = this.battlefield.canvas.offsetWidth;
                const height = this.battlefield.canvas.offsetHeight;
                for (const tankEid of activeTanks) {
                    updateTankWithSharedRL(tankEid, width, height, 1000, this.isTraining);
                }
            }
            // Execute game tick
            this.battlefield.gameTick(TICK_TIME_SIMULATION);

            // Check if episode is done
            const activeCount = getActiveTankCount();
            const isEpisodeDone = activeCount <= 1 || this.frameCount >= MAX_STEPS;

            if (isEpisodeDone) {
                console.log(`Episode ${ this.episodeCount } completed after ${ this.frameCount } frames`);
                console.log(`Surviving tanks: ${ activeCount }`);

                // Log episode completion
                const stats = logEpisodeCompletion(this.frameCount);

                // Save episode stats
                this.episodeStats.add({
                    episodeNumber: this.episodeCount,
                    duration: this.frameCount,
                    avgReward: stats.avgReward,
                    survivingTanks: activeCount,
                    epsilon: stats.epsilon,
                    loss: stats.avgLoss,
                });

                // Save model periodically
                if (this.episodeCount % getCurrentExperiment().saveModelEvery === 0) {
                    this.logStats();
                    saveSharedModel();
                }

                this.resetEnvironment();
            }
        }, TICK_TIME_REAL);
    }

    // Log statistics
    private logStats() {
        const last10Episodes = this.episodeStats.toArray();

        const avgDuration = last10Episodes.reduce((sum, stat) => sum + stat.duration, 0) / Math.max(1, last10Episodes.length);
        const avgReward = last10Episodes.reduce((sum, stat) => sum + stat.avgReward, 0) / Math.max(1, last10Episodes.length);
        const avgSurvivors = last10Episodes.reduce((sum, stat) => sum + stat.survivingTanks, 0) / Math.max(1, last10Episodes.length);
        const avgEpsilon = last10Episodes.reduce((sum, stat) => sum + stat.epsilon, 0) / Math.max(1, last10Episodes.length);
        const avgLoss = last10Episodes.reduce((sum, stat) => sum + (stat.loss || 0), 0) / Math.max(1, last10Episodes.length);

        // Calculate average frame time
        const avgFrameTime = this.frameTimeSamples.length > 0
            ? this.frameTimeSamples.reduce((a, b) => a + b, 0) / this.frameTimeSamples.length
            : 0;

        // Calculate average training time
        const avgTrainingTime = this.trainingTimeSamples.length > 0
            ? this.trainingTimeSamples.reduce((a, b) => a + b, 0) / this.trainingTimeSamples.length
            : 0;

        console.log('====== TRAINING STATISTICS ======');
        console.log(`Episodes completed: ${ this.episodeCount }`);
        console.log(`Average duration (last 10): ${ avgDuration.toFixed(1) } frames`);
        console.log(`Average reward (last 10): ${ avgReward.toFixed(2) }`);
        console.log(`Average survivors (last 10): ${ avgSurvivors.toFixed(1) } tanks`);
        console.log(`Current epsilon: ${ avgEpsilon.toFixed(4) }`);
        console.log(`Average loss (last 10): ${ avgLoss.toFixed(4) }`);
        console.log(`Average frame time: ${ avgFrameTime.toFixed(2) }ms`);
        console.log(`Average training time: ${ avgTrainingTime.toFixed(2) }ms`);
        console.log(`Current experiment: ${ getCurrentExperiment().name }`);
        console.log('================================');
    }
}
