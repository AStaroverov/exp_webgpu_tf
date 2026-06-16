/**
 * dashboard — the lil-gui control/info panel, adapted from ppo_unknown's
 * `ui/debug.ts` (createDebugVisualization). Trimmed to what the single-thread Burn
 * loop exposes: Render (P) / Greedy (G) / Charts (M) toggles plus a live Info readout
 * (iteration version, last success, per-agent recent reward). The charts toggle drives
 * our `metricsPanel` (same M keybind / behaviour as the real dashboard).
 *
 * Coupling removed vs the original: no Download/Reset of tfjs IndexedDB models, no
 * `getDrawState` Dexie persistence — the toggles live in memory and drive the
 * in-process loop directly (see index.ts). Greedy flows to `setGreedyInference` on the
 * Burn trainer, the analogue of UnknownAgent.setGreedyInference.
 */

import GUI from "lil-gui";
import { frameTasks } from "../../../lib/TasksScheduler/frameTasks.ts";
import { GameDI } from "../../unknown/src/Game/DI/GameDI.ts";
import { getGameComponents } from "../../unknown/src/Game/ECS/createGameWorld.ts";
import { getTankTeamId } from "../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts";
import { setGreedyInference } from "./trainer.ts";
import { toggleChartsPanel, updateCharts } from "./metricsPanel.ts";
import type { BurnVisManager } from "./BurnVisManager.ts";

export type DashboardControls = {
  render: boolean;
  greedy: boolean;
};

export function createDashboard(
  container: HTMLElement,
  manager: BurnVisManager,
  getStatus: () => { version: number },
): DashboardControls {
  const controls = {
    render: false,
    greedy: false,
    charts: toggleChartsPanel,
  };

  const panel = document.createElement("div");
  Object.assign(panel.style, {
    position: "fixed",
    right: "0",
    top: "0",
    maxHeight: "100vh",
    overflowY: "auto",
    zIndex: "1000",
    display: "flex",
    flexDirection: "column",
  });
  container.appendChild(panel);

  const gui = new GUI({ title: "Burn RL Dashboard", width: 300, autoPlace: false });
  panel.appendChild(gui.domElement);

  const cf = gui.addFolder("Controls");
  cf.add(controls, "render").name("Render (P)");
  cf.add(controls, "greedy").name("Greedy (G)").onChange((v: boolean) => setGreedyInference(v ? true : undefined));
  cf.add(controls, "charts").name("Charts (M)");

  const info = { version: 0, success: 0 };
  const inf = gui.addFolder("Info");
  inf.add(info, "version").name("Version (iters)").disable();
  inf.add(info, "success").name("Success").disable();

  const agentsDiv = document.createElement("div");
  Object.assign(agentsDiv.style, { fontFamily: "monospace", fontSize: "11px", padding: "4px 0" });
  inf.$children.appendChild(agentsDiv);

  document.addEventListener("keypress", (e) => {
    if (e.code === "KeyM") toggleChartsPanel();
    if (e.code === "KeyP") {
      controls.render = !controls.render;
      cf.controllers.forEach((c) => c.updateDisplay());
    }
    if (e.code === "KeyG") {
      controls.greedy = !controls.greedy;
      setGreedyInference(controls.greedy ? true : undefined);
      cf.controllers.forEach((c) => c.updateDisplay());
    }
  });

  frameTasks.addInterval(() => {
    if (!gui.domElement.isConnected) return;
    info.version = getStatus().version;
    info.success = parseFloat(manager.getSuccessRatio().toFixed(2));
    inf.controllers.forEach((c) => c.updateDisplay());
    agentsDiv.innerHTML = getAgentsDebug(manager);
  }, 10);
  frameTasks.addInterval(updateCharts, 30);

  return controls;
}

function getAgentsDebug(manager: BurnVisManager): string {
  if (!GameDI.world) return "";
  const { Color } = getGameComponents(GameDI.world);
  let result = "";
  for (const agent of manager.getAgents()) {
    const eid = agent.tankEid;
    const teamId = getTankTeamId(eid);
    const r = (Color.getR(eid) * 255) | 0;
    const g = (Color.getG(eid) * 255) | 0;
    const b = (Color.getB(eid) * 255) | 0;
    const a = Color.getA(eid);
    result += `<div style="background:rgba(${r},${g},${b},${a});padding:4px;margin:2px 0">
      <div>Tank ${eid} | Team ${teamId}</div>
      <div>Reward: ${agent.getRecentReward().toFixed(2)}</div>
    </div>`;
  }
  return result;
}
