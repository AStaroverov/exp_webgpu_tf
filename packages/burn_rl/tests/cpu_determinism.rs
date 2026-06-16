//! CPU-backend (NdArray) determinism gate.
//!
//! The deterministic oracle for the v4 forward pass: NdArray has no GPU
//! floating-point non-associativity and no autotune, so any divergence here is
//! a pure model-logic bug. This test pinned down the lazy-`Param`-clone
//! re-randomisation root cause (two independent `model.clone()`s drew fresh RNG
//! for every weight) and now guards against its regression: two clones of one
//! fixed model, forwarded on the same input, must be bit-identical.
//!
//! Runs natively only (NdArray is a dev-dependency); no GPU required.
#![cfg(not(target_arch = "wasm32"))]

use burn::module::Module;
use burn::tensor::Tensor;

use burn_rl::test_support::{ActorCriticV4, BOARD_CELLS, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS};

type B = burn_ndarray::NdArray<f32>;

#[test]
fn ndarray_forward_is_deterministic() {
    let device = Default::default();
    let model = ActorCriticV4::<B>::new(&device);

    let n = BOARD_CELLS * BOARD_CHANNELS;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 0.5).collect();
    let board = Tensor::<B, 1>::from_floats(data.as_slice(), &device)
        .reshape([1, BOARD_ROWS, BOARD_COLS, BOARD_CHANNELS]);
    let mask = Tensor::<B, 2>::ones([1, BOARD_CELLS], &device);

    // Two INDEPENDENT clones (the pattern that used to diverge by ~5e-1).
    let (l1, _) = model.clone().forward(board.clone(), mask.clone());
    let (l2, _) = model.clone().forward(board, mask);

    let v1: Vec<f32> = l1.into_data().iter::<f32>().collect();
    let v2: Vec<f32> = l2.into_data().iter::<f32>().collect();
    let max_diff = v1
        .iter()
        .zip(&v2)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);

    assert_eq!(
        max_diff, 0.0,
        "NdArray forward of two clones must be bit-exact (lazy-Param clone regression), got {max_diff:e}"
    );
}
