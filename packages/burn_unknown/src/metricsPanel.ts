/**
 * metricsPanel — the live metrics dashboard (press M), feeding off the SAME
 * `metricsChannels` BroadcastChannels the ppo_unknown dashboard uses. The Burn
 * training loop posts identical payloads (see trainingLoop.publishMetrics), so the
 * charts work unchanged — only the source of the messages differs (one in-process
 * loop instead of learner workers).
 *
 * It reuses ppo_unknown's pure buffer primitives (`RingBuffer`, `CompressedBuffer`)
 * and a Chart.js grid trimmed to the metrics the V4Trainer actually emits
 * (no V-Trace diagnostics / per-head logits, which the learner workers computed and
 * the wasm trainer does not surface). The toggle/keybind contract matches the real
 * dashboard.
 */

import { Chart } from "chart.js/auto";
import { metricsChannels } from "../../ppo/src/infra/channels.ts";
import { scenarioCompositions } from "../../ppo_unknown/src/curriculum/types.ts";
import { CompressedBuffer } from "../../ppo_unknown/src/ui/MetricsBrowser/CompressedBuffer.ts";
import { RingBuffer } from "../../ppo_unknown/src/ui/MetricsBrowser/RingBuffer.ts";

Chart.defaults.color = "#999";
Chart.defaults.borderColor = "#2a2a3e";
Chart.defaults.backgroundColor = "transparent";

type Point = { x: number; y: number };

const scenariosCount = scenarioCompositions.length;

const store = {
  rewards: new CompressedBuffer(10_000, 10),
  values: new RingBuffer(30_000),
  kl: new CompressedBuffer(1_000, 5),
  lr: new CompressedBuffer(1_000, 5),
  policyLoss: new CompressedBuffer(1_000, 5),
  valueLoss: new CompressedBuffer(1_000, 5),
  entropy: new CompressedBuffer(1_000, 5),
  batchSize: new CompressedBuffer(1_000, 5),
  trainTime: new CompressedBuffer(1_000, 5),
  waitTime: new CompressedBuffer(1_000, 5),
  ...Array.from({ length: scenariosCount }, (_, i) => i).reduce(
    (acc, i) => {
      acc[`successRatio${i}`] = new CompressedBuffer(500, 5);
      return acc;
    },
    {} as Record<string, CompressedBuffer>,
  ),
};

// A BroadcastChannel never delivers a message to the SAME object that posted it
// (see ppo/src/infra/channels.ts, the `exit` channel does the same dance). The Burn
// loop posts on `metricsChannels.*` from this very thread, so we must LISTEN on a
// separate channel object with the same name — otherwise `publishMetrics` self-posts
// are silently dropped and the charts never update.
function listenChannel(posted: BroadcastChannel): BroadcastChannel {
  return new BroadcastChannel(posted.name);
}

export function subscribeOnMetrics() {
  const simple: Array<keyof typeof store> = [
    "rewards",
    "values",
    "kl",
    "lr",
    "policyLoss",
    "valueLoss",
    "entropy",
    "batchSize",
    "trainTime",
    "waitTime",
  ];
  for (const key of simple) {
    listenChannel(metricsChannels[key as keyof typeof metricsChannels]).onmessage = (
      e: MessageEvent,
    ) => {
      (store[key] as CompressedBuffer | RingBuffer).add(...(e.data as number[]));
    };
  }
  listenChannel(metricsChannels.successRatio).onmessage = (e: MessageEvent) => {
    (e.data as { scenarioIndex: number; successRatio: number }[]).forEach(
      ({ scenarioIndex, successRatio }) => {
        const k = `successRatio${scenarioIndex}` as keyof typeof store;
        if (k in store) (store[k] as CompressedBuffer).add(successRatio);
      },
    );
  };
}

// ── Panel ────────────────────────────────────────────────────────────────────
let panelEl: HTMLElement | null = null;
let visible = false;
const entries: Array<{ chart: Chart; series: SeriesDef[] }> = [];

interface SeriesDef {
  getData: () => Point[];
  color: string;
  label?: string;
  width?: number;
  dot?: boolean;
}

export function toggleChartsPanel() {
  if (!panelEl) {
    panelEl = buildPanel();
    document.body.appendChild(panelEl);
  }
  visible = !visible;
  panelEl.style.display = visible ? "flex" : "none";
  if (visible) updateCharts();
}

export function updateCharts() {
  if (!visible) return;
  for (const entry of entries) {
    for (let i = 0; i < entry.series.length; i++) {
      entry.chart.data.datasets[i].data = entry.series[i].getData();
    }
    entry.chart.update("none");
  }
}

function buildPanel(): HTMLElement {
  const panel = document.createElement("div");
  Object.assign(panel.style, {
    position: "fixed",
    inset: "0",
    background: "rgba(10, 10, 20, 0.95)",
    overflowY: "auto",
    zIndex: "999",
    display: "none",
    flexDirection: "column",
    padding: "12px",
    fontFamily: "monospace",
    color: "#ccc",
  });

  const header = document.createElement("div");
  Object.assign(header.style, { display: "flex", justifyContent: "space-between", marginBottom: "12px" });
  const title = document.createElement("span");
  title.textContent = "Metrics (Burn single-thread loop)";
  title.style.fontSize = "15px";
  const closeBtn = document.createElement("button");
  closeBtn.textContent = "Close (M)";
  Object.assign(closeBtn.style, {
    cursor: "pointer",
    padding: "4px 12px",
    background: "#333",
    color: "#ccc",
    border: "1px solid #555",
    borderRadius: "4px",
    fontFamily: "monospace",
  });
  closeBtn.onclick = toggleChartsPanel;
  header.append(title, closeBtn);
  panel.appendChild(header);

  const grid = document.createElement("div");
  Object.assign(grid.style, {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(420px, 1fr))",
    gap: "8px",
  });
  panel.appendChild(grid);

  section(grid, "Success Ratios");
  for (let i = 0; i < scenariosCount; i++) {
    addChart(grid, `Scenario ${i}`, [
      { getData: () => store[`successRatio${i}` as keyof typeof store].toArray(), color: "#4a9eff", label: "Train", dot: true },
      { getData: () => movingAvg(store[`successRatio${i}` as keyof typeof store].toArray(), 20), color: "#ff6b6b", label: "MA", width: 2 },
    ]);
  }

  section(grid, "Training");
  addChart(grid, "Episode Return (rewards)", [
    { getData: () => store.rewards.toArray(), color: "#4caf50", label: "Reward" },
  ]);
  addChart(grid, "KL", [{ getData: () => store.kl.toArray(), color: "#9c27b0", label: "KL" }]);
  addChart(grid, "Learning Rate", [{ getData: () => store.lr.toArray(), color: "#9e9e9e", label: "LR" }]);
  addChart(grid, "Entropy H(π)", [{ getData: () => store.entropy.toArray(), color: "#03a9f4", label: "Entropy" }]);
  addChart(grid, "Policy Loss", [{ getData: () => store.policyLoss.toArray(), color: "#ff9800", label: "Policy Loss" }]);
  addChart(grid, "Value Loss", [{ getData: () => store.valueLoss.toArray(), color: "#e91e63", label: "Value Loss" }]);
  addChart(grid, "Value", [{ getData: () => store.values.toArray(), color: "#b197fc", label: "Value" }]);
  addChart(grid, "Batch Size", [{ getData: () => store.batchSize.toArray(), color: "#51cf66", label: "Size" }]);

  section(grid, "Timing (s)");
  addChart(grid, "Train Time (s)", [
    { getData: () => store.trainTime.toArray(), color: "#4a9eff", label: "Train" },
    { getData: () => movingAvg(store.trainTime.toArray(), 100), color: "#ff6b6b", label: "MA", width: 2 },
  ]);
  addChart(grid, "Wait Time (s)", [
    { getData: () => store.waitTime.toArray(), color: "#ffa94d", label: "Wait" },
    { getData: () => movingAvg(store.waitTime.toArray(), 100), color: "#ff6b6b", label: "MA", width: 2 },
  ]);

  return panel;
}

function section(container: HTMLElement, text: string) {
  const h = document.createElement("div");
  h.textContent = text;
  Object.assign(h.style, {
    gridColumn: "1 / -1",
    fontSize: "13px",
    fontWeight: "bold",
    color: "#aaa",
    borderBottom: "1px solid #333",
    padding: "8px 0 4px",
  });
  container.appendChild(h);
}

function addChart(container: HTMLElement, title: string, series: SeriesDef[]) {
  const wrapper = document.createElement("div");
  Object.assign(wrapper.style, { background: "#111122", borderRadius: "4px", padding: "4px" });
  const canvas = document.createElement("canvas");
  wrapper.appendChild(canvas);
  container.appendChild(wrapper);

  const chart = new Chart(canvas, {
    type: "line",
    data: {
      datasets: series.map((s) => ({
        data: [] as Point[],
        borderColor: s.color,
        borderWidth: s.dot ? 0 : (s.width ?? 1.2),
        pointRadius: s.dot ? 1.5 : 0,
        backgroundColor: s.color,
        showLine: !s.dot,
        fill: false,
        tension: 0,
        label: s.label ?? "",
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 2,
      animation: false,
      parsing: false,
      scales: {
        x: { type: "linear", ticks: { maxTicksLimit: 6, font: { size: 10 } }, grid: { color: "#1a1a2e" } },
        y: { type: "linear", ticks: { maxTicksLimit: 5, font: { size: 10 } }, grid: { color: "#1a1a2e" } },
      },
      plugins: {
        legend: { display: series.length > 1, labels: { boxWidth: 12, font: { size: 10 } } },
        title: { display: true, text: title, font: { size: 12, weight: "normal" } },
        decimation: { enabled: true, algorithm: "lttb", samples: 800 },
      },
    },
  });
  entries.push({ chart, series });
}

function movingAvg(data: Point[], windowSize: number): Point[] {
  if (data.length === 0) return [];
  const result: Point[] = [];
  let sum = 0;
  const w = Math.min(windowSize, data.length);
  for (let i = 0; i < w; i++) {
    sum += data[i].y;
    result.push({ x: data[i].x, y: sum / (i + 1) });
  }
  for (let i = w; i < data.length; i++) {
    sum += data[i].y - data[i - windowSize].y;
    result.push({ x: data[i].x, y: sum / windowSize });
  }
  return result;
}
