import init, { Trainer, IterStats } from "./pkg/burn_rl.js";

const statusEl = document.getElementById("status")!;
const logEl = document.getElementById("log")!;
const gridEl = document.getElementById("grid") as HTMLTableElement;
const train1Btn = document.getElementById("train1") as HTMLButtonElement;
const train20Btn = document.getElementById("train20") as HTMLButtonElement;

const ARROWS = ["↑", "↓", "←", "→"];

function log(line: string) {
  logEl.textContent = line + "\n" + logEl.textContent;
}

async function main() {
  if (!("gpu" in navigator)) {
    statusEl.textContent = "WebGPU not available in this browser.";
    return;
  }

  // Instantiate the wasm module, then build the trainer (initialises the GPU device).
  await init();
  const trainer = await Trainer.create(5, 64, 42);
  const size = trainer.grid_size;
  const goal = size * size - 1;

  statusEl.textContent = "ready.";
  train1Btn.disabled = false;
  train20Btn.disabled = false;

  // Render the learned greedy policy + value function on the grid.
  async function renderPolicy() {
    const actions = await trainer.greedy_actions();
    const values = await trainer.values();
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;

    gridEl.innerHTML = "";
    for (let r = 0; r < size; r++) {
      const tr = gridEl.insertRow();
      for (let c = 0; c < size; c++) {
        const cell = r * size + c;
        const td = tr.insertCell();
        const t = (values[cell] - min) / span;
        // green-tinted heatmap by value
        td.style.background = `rgb(${Math.round(20 + t * 30)}, ${Math.round(40 + t * 140)}, ${Math.round(40 + t * 60)})`;
        td.textContent = cell === goal ? "★" : ARROWS[actions[cell]];
      }
    }
  }

  async function trainIterations(count: number) {
    train1Btn.disabled = true;
    train20Btn.disabled = true;
    for (let i = 0; i < count; i++) {
      const s: IterStats = await trainer.train_iteration();
      log(
        `iter: ret=${s.avg_return.toFixed(3)} ` +
          `pl=${s.policy_loss.toFixed(3)} vl=${s.value_loss.toFixed(3)} ` +
          `ent=${s.entropy.toFixed(3)} eps=${s.episodes} steps=${s.steps}`,
      );
      // yield to the event loop so the page stays responsive
      await new Promise((res) => setTimeout(res, 0));
    }
    await renderPolicy();
    train1Btn.disabled = false;
    train20Btn.disabled = false;
  }

  train1Btn.onclick = () => trainIterations(1);
  train20Btn.onclick = () => trainIterations(20);

  await renderPolicy();
}

train1Btn.disabled = true;
train20Btn.disabled = true;
main().catch((e) => {
  statusEl.textContent = "error: " + e;
  console.error(e);
});
