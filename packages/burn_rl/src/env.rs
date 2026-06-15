//! A tiny grid-world environment — the RL "hello world".
//!
//! An agent lives on an `N x N` grid and must reach the goal cell (bottom-right
//! corner). Each step costs a small negative reward; reaching the goal yields
//! `+1` and ends the episode. Episodes also time out after a step budget so a
//! stuck agent does not stall the rollout.
//!
//! The observation is a one-hot vector of length `N*N` (the current cell). This
//! keeps the network input trivial and lets a small MLP learn an exact policy.

use crate::rng::Rng;

pub const N_ACTIONS: usize = 4; // up, down, left, right

pub struct GridWorld {
    size: usize,
    pos: usize,        // current cell index, row-major
    goal: usize,       // goal cell index
    steps: usize,      // steps taken in the current episode
    max_steps: usize,  // episode step budget
}

pub struct StepResult {
    pub reward: f32,
    pub done: bool,
}

impl GridWorld {
    pub fn new(size: usize) -> Self {
        let goal = size * size - 1;
        Self {
            size,
            pos: 0,
            goal,
            steps: 0,
            max_steps: size * size * 2,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn obs_dim(&self) -> usize {
        self.size * self.size
    }

    /// Reset to a random non-goal cell and return the new observation.
    pub fn reset(&mut self, rng: &mut Rng) -> Vec<f32> {
        let cells = (self.size * self.size) as u32;
        loop {
            let c = rng.next_below(cells) as usize;
            if c != self.goal {
                self.pos = c;
                break;
            }
        }
        self.steps = 0;
        self.obs()
    }

    /// One-hot observation of the current cell.
    pub fn obs(&self) -> Vec<f32> {
        let mut v = vec![0.0; self.size * self.size];
        v[self.pos] = 1.0;
        v
    }

    /// One-hot observation of an arbitrary cell (used for policy visualisation).
    pub fn obs_of(&self, cell: usize) -> Vec<f32> {
        let mut v = vec![0.0; self.size * self.size];
        v[cell] = 1.0;
        v
    }

    pub fn step(&mut self, action: usize) -> StepResult {
        let row = self.pos / self.size;
        let col = self.pos % self.size;
        let (mut r, mut c) = (row as i32, col as i32);
        match action {
            0 => r -= 1, // up
            1 => r += 1, // down
            2 => c -= 1, // left
            _ => c += 1, // right
        }
        // Clamp to the grid: bumping a wall keeps the agent in place.
        let max = self.size as i32 - 1;
        r = r.clamp(0, max);
        c = c.clamp(0, max);
        self.pos = (r as usize) * self.size + c as usize;
        self.steps += 1;

        if self.pos == self.goal {
            StepResult { reward: 1.0, done: true }
        } else if self.steps >= self.max_steps {
            StepResult { reward: -0.01, done: true }
        } else {
            StepResult { reward: -0.01, done: false }
        }
    }
}
