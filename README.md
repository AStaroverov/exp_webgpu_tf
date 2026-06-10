# exp_webgpu_tf

A personal hobby project — my after-hours playground for the intersection of
**three things I find fun**: building a game engine from scratch, doing real-time
graphics on the GPU, and training agents to play the game with reinforcement
learning. Everything runs **in the browser**, with no backend: the simulation,
the renderer, and the neural-network training all live in the same TypeScript
monorepo and execute across Web Workers.

The real goal is to **push the browser to its absolute limit** — to see how much
a single tab can actually do. Run a physics sim, render it on the GPU, and train
neural networks on it _at the same time_, all client-side, and find where it
breaks. It's a stress test for the platform as much as anything else; not a
product.

## What I'm trying to do

1. **Write my own ECS game engine** — an entity-component-system simulation
   with a 2D physics backend (Rapier), running deterministically and headless
   so it can be stepped as fast as possible for training.
2. **Render it on the GPU directly** — a hand-rolled **WebGPU** renderer with my
   own WGSL shader/struct abstractions, instead of using Three.js or pixi.
3. **Train agents to play it with PPO** — a from-scratch **Proximal Policy
   Optimization** implementation on top of TensorFlow.js, with a distributed
   actor/learner setup and curriculum learning, all inside the browser tab.

The recurring theme: cram all of it — engine, GPU rendering, and RL training —
into one browser tab at once and squeeze every last drop of performance out of
WebGPU, WASM, and Web Workers.

## How a training run is wired

Everything runs client-side, fanned out across Web Workers from a single page

- **Actors** (multiple workers) run the headless game simulation + run policy
  inference to collect experience, on the **WASM** TF backend.
- **Learners** (policy + value workers) train the networks on the **WebGPU** TF
  backend.
- The **main tab** runs a live debug visualizer that renders an episode driven
  by the latest saved policy, plus a realtime metrics dashboard (React +
  HeroUI), reading channels published by the learners.

The RL setup treats a step as a _decision point_ rather than a single game tick
(a semi-MDP / options framing), with action masking for invalid moves and a
curriculum that grows scenario difficulty over the run.

## Tech stack

TypeScript · Web Workers · WebGPU (custom renderer) · WGSL ·
`bitecs` (ECS) · `@dimforge/rapier2d` (physics) ·
TensorFlow.js (`webgpu` + `wasm` backends).

## Status

Active personal experiment — APIs, package names, and the game itself change
freely as I explore. Not intended for outside use; shared mostly as a record of
what I'm tinkering with.
