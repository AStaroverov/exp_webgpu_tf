# burn_rl — PPO in Burn, in the browser

A self-contained **test implementation** of a reinforcement-learning **train loop**
and **inference**, built on the [Burn](https://github.com/tracel-ai/burn) deep-learning
framework (Rust), compiled to **WASM**, and running on the **WebGPU** backend in the
browser.

It is intentionally minimal and separate from the game packages — its job is to prove
that the full RL loop (rollout → advantages → autodiff → optimiser step → inference)
works end-to-end under `burn-wgpu` in wasm.

## What it does

- **Environment** (`src/env.rs`) — a tiny `N×N` grid-world. The agent starts on a random
  cell and must reach the goal (bottom-right). One-hot observation, 4 actions.
- **Model** (`src/model.rs`) — an actor-critic MLP (shared trunk → policy head + value
  head), generic over the Burn `Backend`.
- **Algorithm** (`src/ppo.rs` + `src/lib.rs`) — **PPO**: GAE advantage estimation,
  clipped surrogate objective, value loss, entropy bonus, Adam. Mirrors the algorithm
  used by the repo's `ppo` / `ppo_unknown` packages, but in Rust/Burn.
- **Backend** — `Autodiff<Wgpu>` for training, plain `Wgpu` for inference. All tensor
  reads use `into_data_async().await` (synchronous readbacks deadlock under wasm).

## Layers, mapped to the request

| Request    | Where                                                              |
| ---------- | ------------------------------------------------------------------ |
| Train loop | `Trainer::train_iteration` — collects a rollout, runs PPO epochs   |
| Inference  | `Trainer::infer` / `greedy_actions` / `values`                     |

## Running

Requires a Rust toolchain with the `wasm32-unknown-unknown` target and `wasm-pack`, and a
browser with **WebGPU** enabled.

```bash
cd packages/burn_rl
npm run dev        # builds the wasm (release) then starts vite
# open the printed URL in a WebGPU-capable browser
```

In the page: **Train 1 / Train 20 iterations** runs PPO; the grid then renders the
learned greedy policy (arrows) over a value heat-map, and the log shows per-iteration
average return / losses / entropy.

Scripts:

- `npm run wasm` — `wasm-pack build --release` → `pkg/`
- `npm run wasm:dev` — faster, unoptimised wasm build
- `npm run dev` / `npm run build` / `npm run preview` — Vite, building the wasm first

## Notes / gotchas

- `getrandom` 0.3 needs an explicit wasm backend — see `.cargo/config.toml`
  (`getrandom_backend="wasm_js"`); Burn uses it for parameter init.
- `burn` is pulled with `default-features = false` (no `std`/threads/tokio) and the
  `wgpu` + `autodiff` features. The loop is driven by hand — no `burn-train` `Learner`.
- The WebGPU backend runs on WebGPU only (no WebGL2 fallback).
- Verified to **compile** (`cargo check`) and **build** (`wasm-pack` + `vite build`).
  Actually *running* it requires a real browser with WebGPU — that step is manual.
