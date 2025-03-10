import * as tf from '@tensorflow/tfjs';
import { getCurrentExperiment } from './experiment-config';
import { entityExists, hasComponent } from 'bitecs';
import {
    cleanupAllRL,
    completeAgentEpisode,
    deactivateTank,
    getActiveTankCount,
    initController,
    isActiveTank,
    memorizeTankBehaviour,
    registerTank,
    resetController,
    saveSharedAgent,
    trainSharedModel,
    updateTankBehaviour,
} from './controller.ts';
import { createBattlefield } from '../Common/createBattlefield.ts';
import {
    MAX_STEPS,
    TANK_COUNT_SIMULATION_MAX,
    TANK_COUNT_SIMULATION_MIN,
    TICK_TIME_REAL,
    TICK_TIME_SIMULATION,
} from '../Common/consts.ts';
import { macroTasks } from '../../../../../lib/TasksScheduler/macroTasks.ts';
import { RingBuffer } from 'ring-buffer-ts';
import { Tank } from '../../ECS/Components/Tank.ts';
import { DI } from '../../DI';
import { randomRangeInt } from '../../../../../lib/random.ts';

let sharedRLGameManager: SharedRLGameManager | null = null;

export function getRLGameManger(): SharedRLGameManager {
    if (!sharedRLGameManager) {
        sharedRLGameManager = new SharedRLGameManager();
    }
    return sharedRLGameManager;
}


// Main class to manage PPO reinforcement learning integration with the game
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
        policyLoss: number;
        valueLoss: number;
    }>(10);
    private stopFrameInterval: VoidFunction | null = null;
    private stopTrainingInterval: VoidFunction | null = null;

    constructor(isTraining: boolean = true) {
        this.isTraining = isTraining;
        console.log(`SharedRLGameManager initialized in ${ isTraining ? 'training' : 'evaluation' } mode`);
        console.log(`Using experiment: ${ getCurrentExperiment().name }`);
    }

    getEpisodeCount() {
        return this.episodeCount;
    }

    // Initialize the game environment
    async init() {
        console.log('Initializing shared tank PPO game environment...');

        // Ensure TensorFlow.js is ready
        await tf.ready();
        console.log('TensorFlow.js ready');

        // Initialize shared RL controller
        const agent = initController(!this.isTraining);

        // Load trained model if requested
        try {
            await agent.load();
            this.loadState();
        } catch (error) {
            console.error('Error loading PPO models:', error);
        }

        // Initialize battlefield with tanks
        this.resetEnvironment();

        return this;
    }

    // Reset environment for a new episode
    resetEnvironment() {
        resetController();

        // Create new battlefield
        this.battlefield?.destroy();
        this.battlefield = createBattlefield(randomRangeInt(TANK_COUNT_SIMULATION_MIN, TANK_COUNT_SIMULATION_MAX));

        // Register each tank with the RL system
        for (const tankEid of this.battlefield.tanks) {
            registerTank(tankEid);
        }

        // Reset frame counter
        this.frameCount = 0;
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

        console.log('Starting shared tank PPO game loop...');
        this.gameLoopRunning = true;
        this.gameLoop();

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

        console.log('Shared tank PPO game loop stopped');

        return this;
    }

    // Save models
    async save() {
        try {
            return (await saveSharedAgent()) && this.saveState();
        } catch (error) {
            console.error('Error saving PPO models:', error);
            return false;
        }
    }

    saveState() {
        try {
            localStorage.setItem('tank-rl-manager-state', JSON.stringify({
                episodeCount: this.episodeCount,
            }));
            console.log('PPO manager state saved');
            return true;
        } catch (error) {
            console.error('Error saving PPO manager state:', error);
            return false;
        }
    }

    loadState() {
        try {
            const state = localStorage.getItem('tank-rl-manager-state');
            if (!state) {
                console.warn('No saved PPO manager state found');
                return false;
            }

            const { episodeCount } = JSON.parse(state);
            this.episodeCount = episodeCount;
            console.log('PPO manager state loaded');
            return true;
        } catch (error) {
            console.error('Error loading PPO manager state:', error);
            return false;
        }
    }

    // Clean up resources
    cleanup() {
        this.stop();
        cleanupAllRL();
        console.log('Shared tank PPO game manager cleaned up');
    }

    // Main game loop
    private gameLoop() {
        let activeTanks: number[];
        this.stopFrameInterval = macroTasks.addInterval(() => {
            if (!this.gameLoopRunning) {
                this.stopFrameInterval?.();
                this.stopFrameInterval = null;
                return;
            }

            // Update frame counter
            this.frameCount++;

            const width = this.battlefield.canvas.offsetWidth;
            const height = this.battlefield.canvas.offsetHeight;
            const isWarmup = this.frameCount < 100;
            const shouldAction = this.frameCount > 0 && this.frameCount % 10 === 0;
            const shouldMemorize = this.frameCount > 3 && (
                (this.frameCount - 3) % 10 === 0
                || (this.frameCount - 5) % 10 === 0
                || (this.frameCount - 7) % 10 === 0
                || (this.frameCount - 9) % 10 === 0
            );
            const isLastMemorize = this.frameCount > 3 && (this.frameCount - 9) % 10 === 0;

            DI.shouldCollectTensor = this.frameCount > 0 && (this.frameCount + 1) % 10 === 0;

            if (shouldAction) {
                activeTanks = this.battlefield.tanks.filter(tankEid => {
                    if (entityExists(this.battlefield.world, tankEid) && hasComponent(this.battlefield.world, tankEid, Tank)) {
                        return true;
                    } else if (isActiveTank(tankEid)) {
                        deactivateTank(tankEid);
                        return false;
                    }
                });

                // Update each tank's RL controller
                for (const tankEid of activeTanks) {
                    updateTankBehaviour(tankEid, width, height, isWarmup);
                }
            }

            // Execute game tick
            this.battlefield.gameTick(TICK_TIME_SIMULATION);

            if (isWarmup) {
                return;
            }

            if (shouldMemorize) {
                for (const tankEid of activeTanks) {
                    memorizeTankBehaviour(tankEid, width, height, this.episodeCount, isLastMemorize ? 0.4 : 0.2, isLastMemorize);
                }
            }

            // Check if episode is done
            const activeCount = getActiveTankCount();
            const isEpisodeDone = activeCount <= 1 || this.frameCount >= MAX_STEPS;

            if (isEpisodeDone) {
                this.stopFrameInterval?.();
                this.stopFrameInterval = null;

                console.log(`Episode ${ this.episodeCount } completed after ${ this.frameCount } frames`);
                console.log(`Surviving tanks: ${ activeCount }`);

                const promiseTrain = trainSharedModel(this.episodeCount);

                // Log episode completion
                const stats = completeAgentEpisode(this.frameCount);

                // Save episode stats
                this.episodeStats.add({
                    episodeNumber: this.episodeCount,
                    duration: this.frameCount,
                    avgReward: stats.avgReward,
                    survivingTanks: activeCount,
                    policyLoss: stats.avgPolicyLoss,
                    valueLoss: stats.avgValueLoss,
                });

                // Save model periodically
                if (this.episodeCount % getCurrentExperiment().saveModelEvery === 0) {
                    promiseTrain.then(({ value, policy }) => {
                        console.log(`Model trained with losses value: ${ value }, policy: ${ policy }`);
                        this.logStats();
                        this.save();
                    });
                }

                this.resetEnvironment();
                this.gameLoop();
            }
        }, TICK_TIME_REAL);
    }

    // Log statistics
    private logStats() {
        const last10Episodes = this.episodeStats.toArray();

        const avgDuration = last10Episodes.reduce((sum, stat) => sum + stat.duration, 0) / Math.max(1, last10Episodes.length);
        const avgReward = last10Episodes.reduce((sum, stat) => sum + stat.avgReward, 0) / Math.max(1, last10Episodes.length);
        const avgSurvivors = last10Episodes.reduce((sum, stat) => sum + stat.survivingTanks, 0) / Math.max(1, last10Episodes.length);
        const avgPolicyLoss = last10Episodes.reduce((sum, stat) => sum + (stat.policyLoss || 0), 0) / Math.max(1, last10Episodes.length);
        const avgValueLoss = last10Episodes.reduce((sum, stat) => sum + (stat.valueLoss || 0), 0) / Math.max(1, last10Episodes.length);

        console.log('====== PPO TRAINING STATISTICS ======');
        console.log(`Current experiment: ${ getCurrentExperiment().name }`);
        console.log(`Episodes completed: ${ this.episodeCount }`);
        console.log(`Average duration (last 10): ${ avgDuration.toFixed(1) } frames`);
        console.log(`Average reward (last 10): ${ avgReward.toFixed(2) }`);
        console.log(`Average survivors (last 10): ${ avgSurvivors.toFixed(1) } tanks`);
        console.log('Average Losses:');
        console.log(`  Policy: ${ avgPolicyLoss.toFixed(3) }`);
        console.log(`  Value: ${ avgValueLoss.toFixed(3) }`);
        console.log('=====================================');
    }
}