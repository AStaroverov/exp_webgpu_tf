import GUI from 'lil-gui';
import { query } from 'bitecs';
import { Color } from 'renderer/src/ECS/Components/Common.ts';
import { frameTasks } from '../../lib/TasksScheduler/frameTasks.ts';
import { VisTestEpisodeManager } from '../ml/src/PPO/VisTest/VisTestEpisodeManager.ts';
import { GameDI } from '../tanks/src/Game/DI/GameDI.ts';
import { Vehicle } from '../tanks/src/Game/ECS/Components/Vehicle.ts';
import { TeamRef } from '../tanks/src/Game/ECS/Components/TeamRef.ts';
import { CONFIG } from './config.ts';
import { toggleChartsPanel, updateCharts } from './Metrics/Browser/index.ts';
import { getDrawState, setDrawState, getUseNoise, setUseNoise, resetState, downloadModels, settingsReady } from './uiUtils.ts';
import { Pilot } from '../tanks/src/Plugins/Pilots/Components/Pilot.ts';
import { CurrentActorAgent } from '../tanks/src/Plugins/Pilots/Agents/CurrentActorAgent.ts';

export function createDebugVisualization(container: HTMLElement, manager: VisTestEpisodeManager) {
    const gui = new GUI({ title: 'RL Dashboard', width: 300, autoPlace: false });
    gui.domElement.style.position = 'fixed';
    gui.domElement.style.right = '0';
    gui.domElement.style.top = '0';
    gui.domElement.style.maxHeight = '100vh';
    gui.domElement.style.overflowY = 'auto';
    gui.domElement.style.zIndex = '1000';
    container.appendChild(gui.domElement);

    setupControls(gui);
    setupInfo(gui, manager);

    // keyboard shortcuts
    document.addEventListener('keypress', (e) => {
        if (e.code === 'KeyP') {
            controls.render = !controls.render;
            setDrawState(controls.render);
        }
        if (e.code === 'KeyM') {
            toggleChartsPanel();
        }
    });

    return gui.domElement;
}

// --- Controls ---

const controls = {
    render: false,
    noise: true,
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
        controls.noise = getUseNoise();
        folder.controllers.forEach(c => c.updateDisplay());
    });

    folder.add(controls, 'render').name('Render (P)').onChange((v: boolean) => setDrawState(v));
    folder.add(controls, 'noise').name('Noise').onChange((v: boolean) => setUseNoise(v));
    folder.add(controls, 'charts').name('Charts (M)');
    folder.add(controls, 'downloadModels').name('Download Models');
    folder.add(controls, 'resetState').name('Reset State');
}

// --- Info + Vehicles ---

function setupInfo(gui: GUI, manager: VisTestEpisodeManager) {
    const folder = gui.addFolder('Info');

    const info = {
        workers: CONFIG.workerCount,
        version: 0,
        success: 0,
    };
    folder.add(info, 'workers').name('Workers').disable();
    folder.add(info, 'version').name('Version').disable();
    folder.add(info, 'success').name('Success').disable();

    const vehicleDiv = document.createElement('div');
    vehicleDiv.style.fontFamily = 'monospace';
    vehicleDiv.style.fontSize = '11px';
    vehicleDiv.style.padding = '4px 0';
    folder.$children.appendChild(vehicleDiv);

    function update() {
        if (!gui.domElement.isConnected) return;
        info.version = manager.getVersion();
        info.success = parseFloat(manager.getSuccessRatio().toFixed(2));
        folder.controllers.forEach(c => c.updateDisplay());
        vehicleDiv.innerHTML = getVehicleDebug(manager);
    }

    function onMetricsTick() {
        if (getDrawState()) return;
        updateCharts();
    }

    frameTasks.addInterval(update, 10);
    frameTasks.addInterval(onMetricsTick, 500);
    update();
}

// --- Vehicle Debug ---

function getVehicleDebug(manager: VisTestEpisodeManager): string {
    if (!GameDI.world) return '';

    let result = '';
    const vehicleEids = query(GameDI.world, [Vehicle, Pilot]);

    for (let i = 0; i < vehicleEids.length; i++) {
        const vehicleEid = vehicleEids[i];
        const pilot = Pilot.getAgent(vehicleEid);
        if (!(pilot instanceof CurrentActorAgent)) continue;

        const teamId = TeamRef.id[vehicleEid];
        const r = (Color.getR(vehicleEid) * 255) | 0;
        const g = (Color.getG(vehicleEid) * 255) | 0;
        const b = (Color.getB(vehicleEid) * 255) | 0;
        const a = Color.getA(vehicleEid);

        result += `<div style="background:rgba(${r},${g},${b},${a});padding:4px;margin:2px 0">
            <div>Vehicle ${vehicleEid} | Team: ${teamId}</div>
            <div>Reward: ${manager.getRecentReward(vehicleEid).toFixed(2)} / ${manager.getDiscounterReward(vehicleEid).toFixed(2)}</div>
            <div>Score: ${manager.getPositiveReward(vehicleEid).toFixed(2)} / ${manager.getNegativeReward(vehicleEid).toFixed(2)}</div>
            <div style="font-size:10px">${Object.entries(manager.getAllRewards(vehicleEid)).map(([m, v]) => `${m}: ${v.toFixed(2)}`).join('<br>')}</div>
        </div>`;
    }

    return result;
}
