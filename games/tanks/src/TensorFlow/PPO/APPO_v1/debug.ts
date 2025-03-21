// DebugInfo singleton to track statistics
import { MasterManager } from './Master/MasterManager.ts';

// Generate debug visualization using HTML and CSS
export function createDebugVisualization(container: HTMLElement, manager: MasterManager) {
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

    // Add stats container
    const statsContainer = document.createElement('div');
    statsContainer.id = 'rl-stats';
    debugContainer.appendChild(statsContainer);

    // Add the debug container to the provided container
    container.appendChild(debugContainer);

    // Update function
    function updateDebugInfo() {
        if (!debugContainer.isConnected) return;

        const agentStats = manager.agent.getStats();
        const memoryUsage = (performance as any).memory.usedJSHeapSize / (1024 * 1024);

        statsContainer.innerHTML = `
            <div>Trains: ${ agentStats.version }</div>
            <div>10 Avg Reward: ${ agentStats.avgReward10.toFixed(2) }</div>
            <div>10 Avg policy loss ${ agentStats.avgPolicyLoss10.toFixed(4) } </div>
            <div>10 Avg value loss ${ agentStats.avgValueLoss10.toFixed(4) } </div>
            <div>Last Avg Reward: ${ agentStats.avgRewardLast?.toFixed(2) }</div>
            <div>Last Avg policy loss ${ agentStats.avgPolicyLossLast?.toFixed(4) } </div>
            <div>Last Avg value loss ${ agentStats.avgValueLossLast?.toFixed(4) } </div>
            <div>Memory: ${ memoryUsage.toFixed(2) }MB</div>
        `;

        // Schedule next update
        requestAnimationFrame(updateDebugInfo);
    }

    // Start updating
    updateDebugInfo();

    return debugContainer;
}