//! Tiny deterministic RNG (xorshift64*). We roll our own so that environment
//! resets and categorical action sampling are reproducible from a seed and need
//! no `getrandom` call on the hot path (getrandom is reserved for one-off
//! parameter initialisation inside burn).

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the zero fixed-point of xorshift.
        Self { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // Top 24 bits → mantissa precision of f32.
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    /// Uniform integer in [0, n).
    pub fn next_below(&mut self, n: u32) -> u32 {
        (self.next_u64() % n as u64) as u32
    }

    /// Sample an index from a (already-normalised) probability slice.
    pub fn sample_categorical(&mut self, probs: &[f32]) -> usize {
        let r = self.next_f32();
        let mut acc = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if r < acc {
                return i;
            }
        }
        probs.len() - 1
    }
}
