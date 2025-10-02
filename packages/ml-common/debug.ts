// DebugInfo singleton to track statistics
import { query } from 'bitecs';
import { Color } from 'renderer/src/ECS/Components/Common.ts';
import { frameTasks } from '../../lib/TasksScheduler/frameTasks.ts';
import { VisTestEpisodeManager } from '../ml/src/PPO/VisTest/VisTestEpisodeManager.ts';
import { GameDI } from '../tanks/src/Game/DI/GameDI.ts';
import { RigidBodyState } from '../tanks/src/Game/ECS/Components/Physical.ts';
import { Tank } from '../tanks/src/Game/ECS/Components/Tank.ts';
import { TeamRef } from '../tanks/src/Game/ECS/Components/TeamRef.ts';
import { CONFIG } from './config.ts';
import { drawMetrics } from './Metrics/Browser/index.ts';
import { getDrawState } from './uiUtils.ts';

// Generate debug visualization using HTML and CSS
export function createDebugVisualization(container: HTMLElement, manager: VisTestEpisodeManager) {
    // Create main container
    const debugContainer = document.createElement('div');
    debugContainer.className = 'debug-container';
    debugContainer.style.position = 'fixed';
    debugContainer.style.right = '560px';
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

    const getDebugInfo = createCommonDebug(manager);
    const getTanksDebug = createTanksDebug(manager);

    // Update function
    function updateDebugInfo() {
        if (!debugContainer.isConnected) return;

        statsContainer.innerHTML = [
            getDebugInfo(),
            getTanksDebug(),
        ].join('<div>-------------</div>');
    }

    async function updateMetrics() {
        if (getDrawState()) return;
        drawMetrics();
    }

    frameTasks.addInterval(updateDebugInfo, 10);
    frameTasks.addInterval(updateMetrics, 300);

    updateDebugInfo();
    updateMetrics();

    return debugContainer;
}

export function createCommonDebug(manager: VisTestEpisodeManager) {
    return () => {
        return `
            <div>Workers: ${CONFIG.workerCount}</div>
            <div>Version: ${manager.getVersion()}</div>
            <div>Success: ${manager.getSuccessRatio().toFixed(2)}</div>
        `;
    };
}

export function createTanksDebug(manager: VisTestEpisodeManager) {
    return () => {
        if (!GameDI.world) return '';

        let result = '';
        const tanksEids = query(GameDI.world, [Tank, RigidBodyState]);

        for (let i = 0; i < tanksEids.length; i++) {

            const tankEid = tanksEids[i];
            const teamId = TeamRef.id[tankEid];
            const color = `rgba(${Color.r[tankEid] * 255}, ${Color.g[tankEid] * 255}, ${Color.b[tankEid] * 255}, ${Color.a[tankEid]})`;

            result += `
                <div style="background: ${color}; padding: 4px;">
                    <div>Tank ${tankEid}</div>
                    <div>Team: ${teamId}</div>
                    <div>Reward: ${manager.getRecentReward(tankEid).toFixed(2)} / ${manager.getDiscounterReward(tankEid).toFixed(2)}</div>
                </div>
                <br>
            `;
        }

        return result;
    };
}

