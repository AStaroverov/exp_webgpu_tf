// Debug visualization for the Shared Tank RL system
import { getCurrentExperiment } from './experiment-config';
import { getSharedAgent } from './agent.ts';
import { getRLGameManger } from './manager.ts';

// DebugInfo singleton to track statistics
export const DebugInfo = {
    getStatsSummary() {
        const manager = getRLGameManger();
        const agent = getSharedAgent();
        const agentStats = agent.getStats();

        return {
            episodeCount: manager.getEpisodeCount(),
            avgReward: agentStats.avgReward,
            memoryUsage: (performance as any).memory.usedJSHeapSize / (1024 * 1024),
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
            <div>Memory: ${ stats.memoryUsage.toFixed(2) }MB</div>
        `;

        // Schedule next update
        requestAnimationFrame(updateDebugInfo);
    }

    // Start updating
    updateDebugInfo();

    return debugContainer;
}