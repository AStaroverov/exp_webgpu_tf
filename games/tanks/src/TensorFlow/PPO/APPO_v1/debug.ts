// DebugInfo singleton to track statistics
import { MasterManager } from './Master/MasterManager.ts';
import { query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { Tank } from '../../../ECS/Components/Tank.ts';
import { RigidBodyState } from '../../../ECS/Components/Physical.ts';
import { hypot } from '../../../../../../lib/math.ts';
import { Color } from '../../../../../../src/ECS/Components/Common.ts';
import { getDrawState } from '../../Common/utils.ts';
import { frameTasks } from '../../../../../../lib/TasksScheduler/frameTasks.ts';
import { CONFIG } from '../Common/config.ts';

// Generate debug visualization using HTML and CSS
export function createDebugVisualization(container: HTMLElement, manager: MasterManager) {
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

    frameTasks.addInterval(updateDebugInfo, 10);

    return debugContainer;
}

export function createCommonDebug(manager: MasterManager) {
    return () => {
        const memoryUsage = (performance as any).memory.usedJSHeapSize / (1024 * 1024);

        return `
            <div>Workers: ${ CONFIG.workerCount }</div>
            <div>Memory: ${ memoryUsage.toFixed(2) }MB</div>
            <div>Version: ${ manager.agent.getVersion() } </div>
        `;
    };
}

export function createTanksDebug(manager: MasterManager) {
    return () => {
        if (!getDrawState()) return '';

        let result = '';
        const tanksEids = query(GameDI.world, [Tank, RigidBodyState]);

        for (let i = 0; i < tanksEids.length; i++) {
            const tankEid = tanksEids[i];
            const position = Array.from(RigidBodyState.position.getBatch(tankEid)).map(v => v.toFixed(2)).join(', ');
            const speed = hypot(RigidBodyState.linvel.get(tankEid, 0), RigidBodyState.linvel.get(tankEid, 1)).toFixed(2);
            const aim = Tank.aimEid[tankEid];
            const color = `rgba(${ Color.r[aim] * 255 }, ${ Color.g[aim] * 255 }, ${ Color.b[aim] * 255 }, ${ Color.a[aim] })`;

            result += `
                <div style="background: ${ color }; padding: 4px;">
                    <div>Tank ${ tankEid }</div>
                    <div>Reward: ${ manager.getReward(tankEid).toFixed(2) }</div>
                    <div>Position: ${ position }</div>
                    <div>Speed: ${ speed }</div>
                </div>
                <br>
            `;
        }

        return result;
    };
}

