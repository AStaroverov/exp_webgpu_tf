/**
 * Debug dashboard for ppo_unknown — the lil-gui control/info panel + the realtime
 * charts panel (MetricsBrowser). Adapted from tanks' `ui/debug.ts`, trimmed to the
 * MVP: controls (charts toggle, download, reset), live version/success info, and a
 * per-agent recent-reward readout. Charts read the shared `metricsChannels` the
 * learners already publish to, so they update in realtime regardless of rendering.
 */

import GUI from 'lil-gui';
import { frameTasks } from '../../../../lib/TasksScheduler/frameTasks.ts';
import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';
import { getTankTeamId } from '../../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { CONFIG } from '../config.ts';
import { UnknownVisTestEpisodeManager } from '../agents/UnknownVisTestEpisodeManager.ts';
import { createLightingGUI } from '../../../unknown/src/ui/createLightingGUI.ts';
import { toggleChartsPanel, updateCharts } from './MetricsBrowser/index.ts';
import { downloadModels, getDrawState, resetState, setDrawState, settingsReady } from './uiUtils.ts';

export function createDebugVisualization(container: HTMLElement, manager: UnknownVisTestEpisodeManager) {
    const gui = new GUI({ title: 'RL Dashboard', width: 300, autoPlace: false });
    Object.assign(gui.domElement.style, {
        position: 'fixed', right: '0', top: '0', maxHeight: '100vh', overflowY: 'auto', zIndex: '1000',
    });
    container.appendChild(gui.domElement);

    const lighting = createLightingGUI({ container, side: 'left' });

    setupControls(gui);
    setupInfo(gui, manager, lighting.sync);

    document.addEventListener('keypress', (e) => {
        if (e.code === 'KeyM') toggleChartsPanel();
        if (e.code === 'KeyP') {
            controls.render = !controls.render;
            setDrawState(controls.render);
        }
    });

    return gui.domElement;
}

const controls = {
    render: false,
    charts: toggleChartsPanel,
    downloadModels,
    resetState() {
        if (confirm('Reset all state? This will reload the page.')) {
            resetState();
        }
    },
};

function setupControls(gui: GUI) {
    const folder = gui.addFolder('Controls');

    settingsReady.then(() => {
        controls.render = getDrawState();
        folder.controllers.forEach((c) => c.updateDisplay());
    });

    folder.add(controls, 'render').name('Render (P)').onChange((v: boolean) => setDrawState(v));
    folder.add(controls, 'charts').name('Charts (M)');
    folder.add(controls, 'downloadModels').name('Download Models');
    folder.add(controls, 'resetState').name('Reset State');
}

function setupInfo(gui: GUI, manager: UnknownVisTestEpisodeManager, syncLighting: () => void) {
    const folder = gui.addFolder('Info');

    const info = { workers: CONFIG.workerCount, version: 0, success: 0 };
    folder.add(info, 'workers').name('Workers').disable();
    folder.add(info, 'version').name('Version').disable();
    folder.add(info, 'success').name('Success').disable();

    const agentsDiv = document.createElement('div');
    Object.assign(agentsDiv.style, { fontFamily: 'monospace', fontSize: '11px', padding: '4px 0' });
    folder.$children.appendChild(agentsDiv);

    function update() {
        if (!gui.domElement.isConnected) return;
        syncLighting();
        info.version = manager.getVersion();
        info.success = parseFloat(manager.getSuccessRatio().toFixed(2));
        folder.controllers.forEach((c) => c.updateDisplay());
        agentsDiv.innerHTML = getAgentsDebug(manager);
    }

    frameTasks.addInterval(update, 10);
    frameTasks.addInterval(updateCharts, 30);
    update();
}

function getAgentsDebug(manager: UnknownVisTestEpisodeManager): string {
    if (!GameDI.world) return '';
    const { Color, Spottable } = getGameComponents(GameDI.world);

    let result = '';
    for (const agent of manager.getAgents()) {
        const eid = agent.tankEid;
        const teamId = getTankTeamId(eid);
        const r = (Color.getR(eid) * 255) | 0;
        const g = (Color.getG(eid) * 255) | 0;
        const b = (Color.getB(eid) * 255) | 0;
        const a = Color.getA(eid);
        // "Am I spotted by the enemy" — the unit's single confidence value.
        const spotted = Spottable.getConfidence(eid).toFixed(2);

        result += `<div style="background:rgba(${r},${g},${b},${a});padding:4px;margin:2px 0">
            <div>Tank ${eid} | Team ${teamId}</div>
            <div>Reward: ${agent.getRecentReward().toFixed(2)}</div>
            <div>Spotted: ${spotted}</div>
        </div>`;
    }
    return result;
}
