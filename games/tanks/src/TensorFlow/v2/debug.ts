// Debug visualization for the Shared Tank RL system
import { getCurrentExperiment } from './experiment-config';
import { getSharedAgent } from './agent.ts';

// DebugInfo singleton to track statistics
export const DebugInfo = {
    // Episodes
    episodes: [] as number[],

    // Rewards
    episodeRewards: [] as number[],
    averageRewards: [] as number[],

    // Training metrics
    episodeLengths: [] as number[],
    losses: [] as number[],
    epsilons: [] as number[],

    // Game metrics
    survivingTanks: [] as number[],
    activeTanks: [] as number[],

    // Performance metrics
    frameTimes: [] as number[],
    trainingTimes: [] as number[],
    memoryUsage: [] as number[],

    // Actions per tank
    tankActions: new Map<number, {
        shoots: number[],
        moves: number[][],
        aims: number[][]
    }>(),

    // Reward components per tank
    tankRewards: new Map<number, {
        health: number[],
        aim: number[],
        movement: number[],
        survival: number[]
    }>(),

    // Add episode data
    addEpisodeData(
        episode: number,
        reward: number,
        length: number,
        survivors: number,
        active: number,
        epsilon: number,
        loss: number,
    ) {
        // Add data to arrays
        this.episodes.push(episode);
        this.episodeRewards.push(reward);
        this.episodeLengths.push(length);
        this.survivingTanks.push(survivors);
        this.activeTanks.push(active);
        this.epsilons.push(epsilon);

        if (loss !== undefined) {
            this.losses.push(loss);
        }

        // Calculate average reward (last 10 episodes)
        const last10Rewards = this.episodeRewards.slice(-10);
        const avgReward = last10Rewards.reduce((a, b) => a + b, 0) / last10Rewards.length;
        this.averageRewards.push(avgReward);

        // Limit array sizes to prevent memory issues
        const MAX_SIZE = 1000;
        if (this.episodes.length > MAX_SIZE) {
            this.episodes.shift();
            this.episodeRewards.shift();
            this.episodeLengths.shift();
            this.survivingTanks.shift();
            this.activeTanks.shift();
            this.epsilons.shift();
            this.averageRewards.shift();

            if (this.losses.length > MAX_SIZE) {
                this.losses.shift();
            }
        }
    },

    // Add performance data
    addPerformanceData(frameTime: number, trainingTime: number) {
        this.frameTimes.push(frameTime);
        this.trainingTimes.push(trainingTime);

        // Track memory usage if available
        // @ts-ignore
        if (performance && performance.memory) {
            this.memoryUsage.push((performance as any).memory.usedJSHeapSize / (1024 * 1024));
        }

        // Limit array sizes
        const MAX_SIZE = 1000;
        if (this.frameTimes.length > MAX_SIZE) {
            this.frameTimes.shift();
            this.trainingTimes.shift();
            if (this.memoryUsage.length > MAX_SIZE) {
                this.memoryUsage.shift();
            }
        }
    },

    // Add tank action data
    addTankAction(tankId: number, action: number[]) {
        if (!this.tankActions.has(tankId)) {
            this.tankActions.set(tankId, {
                shoots: [],
                moves: [],
                aims: [],
            });
        }

        const tankData = this.tankActions.get(tankId)!;
        const [shoot, moveX, moveY, aimX, aimY] = action;

        tankData.shoots.push(shoot);
        tankData.moves.push([moveX, moveY]);
        tankData.aims.push([aimX, aimY]);

        // Limit array sizes
        const MAX_SIZE = 100;
        if (tankData.shoots.length > MAX_SIZE) {
            tankData.shoots.shift();
            tankData.moves.shift();
            tankData.aims.shift();
        }
    },

    // Add tank reward components
    addTankReward(tankId: number, rewardComponents: any) {
        if (!this.tankRewards.has(tankId)) {
            this.tankRewards.set(tankId, {
                health: [],
                aim: [],
                movement: [],
                survival: [],
            });
        }

        const tankData = this.tankRewards.get(tankId)!;

        tankData.health.push(rewardComponents.health || 0);
        tankData.aim.push(rewardComponents.aim || 0);
        tankData.movement.push(rewardComponents.movement || 0);
        tankData.survival.push(rewardComponents.survival || 0);

        // Limit array sizes
        const MAX_SIZE = 100;
        if (tankData.health.length > MAX_SIZE) {
            tankData.health.shift();
            tankData.aim.shift();
            tankData.movement.shift();
            tankData.survival.shift();
        }
    },

    // Clear all data
    clear() {
        this.episodes = [];
        this.episodeRewards = [];
        this.averageRewards = [];
        this.episodeLengths = [];
        this.losses = [];
        this.epsilons = [];
        this.survivingTanks = [];
        this.activeTanks = [];
        this.frameTimes = [];
        this.trainingTimes = [];
        this.memoryUsage = [];
        this.tankActions.clear();
        this.tankRewards.clear();
    },

    // Get current stats summary
    getStatsSummary() {
        const agent = getSharedAgent();
        const agentStats = agent.getStats();

        // Calculate averages
        const avgReward = this.episodeRewards.length > 0
            ? this.episodeRewards.slice(-10).reduce((a, b) => a + b, 0) /
            Math.min(10, this.episodeRewards.length)
            : 0;

        const avgEpisodeLength = this.episodeLengths.length > 0
            ? this.episodeLengths.slice(-10).reduce((a, b) => a + b, 0) /
            Math.min(10, this.episodeLengths.length)
            : 0;

        const avgFrameTime = this.frameTimes.length > 0
            ? this.frameTimes.slice(-100).reduce((a, b) => a + b, 0) /
            Math.min(100, this.frameTimes.length)
            : 0;

        const avgTrainingTime = this.trainingTimes.length > 0
            ? this.trainingTimes.slice(-100).reduce((a, b) => a + b, 0) /
            Math.min(100, this.trainingTimes.length)
            : 0;

        return {
            episodeCount: this.episodes.length > 0 ? this.episodes[this.episodes.length - 1] : 0,
            avgReward,
            avgEpisodeLength,
            avgFrameTime,
            avgTrainingTime,
            memoryUsage: this.memoryUsage.length > 0 ? this.memoryUsage[this.memoryUsage.length - 1] : 0,
            epsilon: agentStats.epsilon,
            experimentName: getCurrentExperiment().name,
            memorySize: agentStats.memorySize,
        };
    },
};

// Generate debug visualization using HTML and CSS
export function createDebugVisualization(container: HTMLElement) {
    // Create main container
    const debugContainer = document.createElement('div');
    debugContainer.className = 'debug-container';
    debugContainer.style.position = 'fixed';
    debugContainer.style.right = '10px';
    debugContainer.style.top = '10px';
    debugContainer.style.width = '300px';
    debugContainer.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
    debugContainer.style.color = 'white';
    debugContainer.style.padding = '10px';
    debugContainer.style.borderRadius = '5px';
    debugContainer.style.fontFamily = 'monospace';
    debugContainer.style.fontSize = '12px';
    debugContainer.style.zIndex = '1000';

    // Add title
    const title = document.createElement('h3');
    title.textContent = 'Shared RL Debug';
    title.style.margin = '0 0 10px 0';
    title.style.textAlign = 'center';
    debugContainer.appendChild(title);

    // Add stats container
    const statsContainer = document.createElement('div');
    statsContainer.id = 'rl-stats';
    debugContainer.appendChild(statsContainer);

    // Add the debug container to the provided container
    container.appendChild(debugContainer);

    // Update function
    function updateDebugInfo() {
        if (!debugContainer.isConnected) return;

        const stats = DebugInfo.getStatsSummary();
        statsContainer.innerHTML = `
            <div>Episode: ${ stats.episodeCount }</div>
            <div>Avg Reward: ${ stats.avgReward.toFixed(2) }</div>
            <div>Epsilon: ${ stats.epsilon.toFixed(4) }</div>
            <div>Memory Size: ${ stats.memorySize }</div>
            <div>Experiment: ${ stats.experimentName }</div>
            <div>Frame Time: ${ stats.avgFrameTime.toFixed(2) }ms</div>
            <div>Train Time: ${ stats.avgTrainingTime.toFixed(2) }ms</div>
            <div>Memory: ${ stats.memoryUsage.toFixed(2) }MB</div>
        `;

        // Schedule next update
        requestAnimationFrame(updateDebugInfo);
    }

    // Start updating
    updateDebugInfo();

    return debugContainer;
}