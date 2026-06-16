//! KL-adaptive learning-rate schedule.
//!
//! Mirrors the TS reference (`ppo/src/utils/getDynamicLearningRate.ts` driven by
//! `createPolicyLearnerAgent.ts`):
//!
//! - A fixed-size ring buffer (`kl_history_size = 25`) of per-epoch KL values.
//!   Each epoch may push one or more KL samples (`add(...klList)`), evicting the
//!   oldest entries once full.
//! - The schedule decision uses the **median** of the whole buffer:
//!   `if median > kl_high (0.02): lr *= kl_mult_high (0.95);
//!    elif median < kl_low (0.005): lr *= kl_mult_low (1.05)`.
//! - The result is clamped to `[lr_min, lr_max]`.
//!
//! Median matches `lib/math.ts`: sort ascending, for even length average the two
//! middle elements, for odd length take the middle element.

use std::collections::VecDeque;

/// Rolling KL-tracking LR scheduler.
pub struct KLScheduler {
    history: VecDeque<f64>,
    history_size: usize,
    kl_high: f64,
    kl_low: f64,
    mult_high: f64,
    mult_low: f64,
    lr_min: f64,
    lr_max: f64,
}

impl KLScheduler {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        history_size: usize,
        kl_high: f64,
        kl_low: f64,
        mult_high: f64,
        mult_low: f64,
        lr_min: f64,
        lr_max: f64,
    ) -> Self {
        Self {
            history: VecDeque::with_capacity(history_size),
            history_size,
            kl_high,
            kl_low,
            mult_high,
            mult_low,
            lr_min,
            lr_max,
        }
    }

    /// Push a single KL sample, evicting the oldest if the buffer is full.
    pub fn push(&mut self, kl: f64) {
        if self.history.len() == self.history_size {
            self.history.pop_front();
        }
        self.history.push_back(kl);
    }

    /// Push several KL samples (one epoch's `klList`).
    pub fn extend<I: IntoIterator<Item = f64>>(&mut self, kls: I) {
        for kl in kls {
            self.push(kl);
        }
    }

    /// Median of the current buffer, matching `lib/math.ts::median`.
    ///
    /// Returns `None` when the buffer is empty.
    pub fn median(&self) -> Option<f64> {
        if self.history.is_empty() {
            return None;
        }
        let mut sorted: Vec<f64> = self.history.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        let m = if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        };
        Some(m)
    }

    /// Adjust `cur_lr` based on a single KL value (no buffering), matching the
    /// pure TS `getDynamicLearningRate`. Clamps to `[lr_min, lr_max]`.
    pub fn adjust(&self, kl: f64, cur_lr: f64) -> f64 {
        let mut lr = cur_lr;
        if kl > self.kl_high {
            lr *= self.mult_high;
        } else if kl < self.kl_low {
            lr *= self.mult_low;
        }
        lr.clamp(self.lr_min, self.lr_max)
    }

    /// Push the latest observed KL, then return the (possibly) adjusted LR using
    /// the median of the buffer.
    ///
    /// This is the primary entry point and mirrors the TS learner loop: add the
    /// sample, take the median over the ring buffer, then apply the multiplier
    /// and clamp. With a non-empty buffer (always true after the push) this never
    /// falls back to the unmodified LR.
    pub fn update(&mut self, kl: f64, cur_lr: f64) -> f64 {
        self.push(kl);
        match self.median() {
            Some(m) => self.adjust(m, cur_lr),
            // Unreachable after push, but keep the LR untouched (still clamped)
            // to match "kl != null ? ... : networkLR" semantics defensively.
            None => cur_lr.clamp(self.lr_min, self.lr_max),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sched() -> KLScheduler {
        // kl_history_size=25, kl_high=0.02, kl_low=0.005,
        // kl_mult_high=0.95, kl_mult_low=1.05
        KLScheduler::new(25, 0.02, 0.005, 0.95, 1.05, 1e-6, 10.0)
    }

    #[test]
    fn high_kl_decreases_lr() {
        let s = sched();
        assert!((s.adjust(0.05, 1.0) - 0.95).abs() < 1e-12);
    }

    #[test]
    fn low_kl_increases_lr() {
        let s = sched();
        assert!((s.adjust(0.001, 1.0) - 1.05).abs() < 1e-12);
    }

    #[test]
    fn mid_kl_no_change() {
        let s = sched();
        assert!((s.adjust(0.01, 1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn clamp_applies() {
        let s = KLScheduler::new(25, 0.02, 0.005, 0.95, 1.05, 0.5, 1.0);
        // low kl would push to 1.05 -> clamp to max 1.0
        assert!((s.adjust(0.001, 1.0) - 1.0).abs() < 1e-12);
        // high kl repeatedly below min -> clamp to min 0.5
        assert!((s.adjust(0.05, 0.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn median_even_and_odd() {
        let mut s = sched();
        s.extend([1.0, 3.0, 2.0]);
        assert_eq!(s.median(), Some(2.0));
        s.push(4.0);
        // sorted [1,2,3,4] -> (2+3)/2 = 2.5
        assert_eq!(s.median(), Some(2.5));
    }

    #[test]
    fn ring_buffer_evicts_oldest() {
        let mut s = KLScheduler::new(3, 0.02, 0.005, 0.95, 1.05, 1e-6, 1.0);
        s.extend([1.0, 2.0, 3.0, 4.0]);
        // capacity 3 -> [2,3,4], median 3
        assert_eq!(s.median(), Some(3.0));
        assert_eq!(s.history.len(), 3);
    }

    #[test]
    fn update_uses_median() {
        let mut s = sched();
        // single high sample -> median high -> lr*0.95
        let lr = s.update(0.05, 1.0);
        assert!((lr - 0.95).abs() < 1e-12);
    }
}
