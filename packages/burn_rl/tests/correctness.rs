//! Native correctness tests for the v4 PPO crate — the H1/H2/H3-class gates the
//! finiteness smoke test cannot catch:
//!
//!   1. `record` save/load round-trip — the persistence-format gate: a v4 model
//!      serialised via burn's `BinBytesRecorder` and reloaded into a fresh model
//!      must produce bit-for-bit (≤ f32 epsilon) identical forward outputs.
//!   2. `HexRingGather` structural gate — gathers EXACTLY the 37
//!      `ACTION_CELL_INDEXES` cells (would have caught the zeroed-index bug).
//!   3. Masked-softmax gate — forbidden actions ≈ 0, distribution sums to 1, Hold
//!      always sampleable even when everything else is masked, mostly-masked row
//!      stays finite.
//!
//! Runnable without a browser (native WebGPU = Metal on macOS):
//!
//! ```sh
//! cargo test --test correctness -- --nocapture
//! ```
//!
//! NOTE on a tfjs forward-parity dump: a *true* numerical parity test against the
//! TF.js `v4.ts` reference is NOT included here — see `tests/README.md` for why
//! (it requires standing up the full tfjs stack with a Burn↔tfjs weight bridge)
//! and the documented procedure to produce such a dump. We do not fake it.

#![cfg(not(target_arch = "wasm32"))]

use burn::backend::wgpu::WgpuDevice;
use burn::module::Module;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::{Tensor, TensorData};

use burn_rl::test_support::{
    masked_softmax, ActorCriticV4, HexRingGather, ACTION_CELL_INDEXES, ACTION_CELL_INDEXES_LEN,
    ACTION_DIM_TOTAL, BOARD_CELLS, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS, HOLD_ACTION, MASK_NEG,
    TestBackend,
};

type B = TestBackend;

fn read<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().iter::<f32>().collect()
}

/// Deterministic fixed board `[1, 11, 11, 22]` for forward-parity assertions.
fn fixed_board(device: &WgpuDevice) -> Tensor<B, 4> {
    let n = BOARD_CELLS * BOARD_CHANNELS;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.013).sin() * 0.5))
        .collect();
    Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([1, BOARD_ROWS, BOARD_COLS, BOARD_CHANNELS])
}

// ===========================================================================
// 1. record save/load round-trip (persistence-format gate).
// ===========================================================================
//
// The gate compares the SERIALISED PARAMETERS, not forward outputs. Comparing
// the recorder's own bytes is the stronger and exact persistence assertion: it
// proves load_record fully overwrote model_b's weights with model_a's,
// parameter-for-parameter, to the bit. (Historical note: this avoided the
// lazy-`Param`-clone non-determinism — now fixed, see `forward_is_deterministic`.)

#[test]
fn record_round_trip_restores_all_parameters() {
    let device = WgpuDevice::default();
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();

    // Two independently-initialised models.
    let model_a = ActorCriticV4::<B>::new(&device);
    let model_b = ActorCriticV4::<B>::new(&device);

    let bytes_a = recorder
        .record(model_a.clone().into_record(), ())
        .expect("record model_a failed");
    let bytes_b0 = recorder
        .record(model_b.clone().into_record(), ())
        .expect("record model_b failed");

    // Non-vacuous: the two random models must serialise differently.
    assert_ne!(
        bytes_a, bytes_b0,
        "two freshly-initialised models serialised identically; the round-trip test would be vacuous"
    );

    // Round-trip model_a's weights INTO model_b.
    let record_a = recorder
        .load(bytes_a.clone(), &device)
        .expect("load into record failed");
    let model_b = model_b.load_record(record_a);

    // After loading, model_b must serialise byte-for-byte identically to model_a:
    // every parameter tensor was restored exactly.
    let bytes_b1 = recorder
        .record(model_b.into_record(), ())
        .expect("re-record model_b failed");

    assert_eq!(
        bytes_a.len(),
        bytes_b1.len(),
        "serialised length changed after round-trip"
    );
    assert!(
        bytes_a == bytes_b1,
        "model_b did not round-trip to model_a's exact weights (serialised bytes differ)"
    );
    eprintln!(
        "record round-trip: {} bytes restored exactly (bit-for-bit)",
        bytes_a.len()
    );
}

/// Forward determinism gate (previously the `forward_is_nondeterministic_known_issue`
/// pin). Two forwards of the SAME fixed model on the WebGPU backend, on the same
/// all-ones-masked input, must be bit-identical.
///
/// ROOT CAUSE that this used to expose (now fixed): Burn `Param`s built by
/// `LinearConfig::init` are *lazy* — a deferred RNG initializer, not a concrete
/// tensor. `Param::clone` of a lazy param duplicates the initializer (burn-core
/// `param/base.rs`: "Each clone holds its own `Uninitialized` state"), so each
/// independent `model.clone()` re-ran the RNG and produced DIFFERENT weights —
/// the two forwards diverged by ~1e-1. It was NEVER GPU non-associativity or
/// autotune (NdArray on CPU diverged identically; see `ndarray_forward_is_deterministic`).
/// `ActorCriticV4::new` now force-materialises all params, so clones copy values.
#[test]
fn forward_is_deterministic() {
    let device = WgpuDevice::default();
    let model = ActorCriticV4::<B>::new(&device);
    let board = fixed_board(&device);
    let board_mask = Tensor::<B, 2>::ones([1, BOARD_CELLS], &device);

    let (l1, _) = model.clone().forward(board.clone(), board_mask.clone());
    let (l2, _) = model.clone().forward(board, board_mask);
    let (l1, l2) = (read(l1), read(l2));

    for (k, v) in l1.iter().chain(l2.iter()).enumerate() {
        assert!(v.is_finite(), "forward produced non-finite logit at {k}: {v}");
    }
    let max_diff = l1
        .iter()
        .zip(&l2)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        max_diff, 0.0,
        "two clones of the same model forwarded identically must be bit-exact, got max diff {max_diff:e}"
    );
}

// ===========================================================================
// 2. HexRingGather structural test (zeroed-index gate).
// ===========================================================================

#[test]
fn hex_ring_gather_selects_action_cells() {
    let device = WgpuDevice::default();
    let gather = HexRingGather::<B>::new(&device);

    // Build tokens [1, 121, dim] where EVERY feature of token `c` equals the flat
    // cell index `c`. The gather must then return, per output token `i`, the
    // value `ACTION_CELL_INDEXES[i]` across all dims.
    let dim = 4usize;
    let mut data = Vec::with_capacity(BOARD_CELLS * dim);
    for c in 0..BOARD_CELLS {
        for _ in 0..dim {
            data.push(c as f32);
        }
    }
    let tokens =
        Tensor::<B, 1>::from_floats(data.as_slice(), &device).reshape([1, BOARD_CELLS, dim]);

    let gathered = gather.forward(tokens); // [1, 37, dim]
    assert_eq!(gathered.dims(), [1, ACTION_CELL_INDEXES_LEN, dim]);
    let out = read(gathered);

    for i in 0..ACTION_CELL_INDEXES_LEN {
        let expected = ACTION_CELL_INDEXES[i] as f32;
        for d in 0..dim {
            let got = out[i * dim + d];
            assert!(
                (got - expected).abs() < 1e-5,
                "gather token {i} dim {d}: expected cell index {expected}, got {got}"
            );
        }
    }
    eprintln!(
        "HexRingGather gathered the expected {ACTION_CELL_INDEXES_LEN} cells: {:?}",
        ACTION_CELL_INDEXES
    );
}

// ===========================================================================
// 3. masked softmax gate.
// ===========================================================================

fn softmax_row(logits_row: &[f32], mask_row: &[f32], device: &WgpuDevice) -> Vec<f32> {
    let logits = Tensor::<B, 2>::from_data(
        TensorData::new(logits_row.to_vec(), [1, ACTION_DIM_TOTAL]),
        device,
    );
    let mask = Tensor::<B, 2>::from_data(
        TensorData::new(mask_row.to_vec(), [1, ACTION_DIM_TOTAL]),
        device,
    );
    read(masked_softmax(logits, mask))
}

#[test]
fn masked_softmax_forbidden_near_zero_and_sums_to_one() {
    let device = WgpuDevice::default();

    // Distinct logits so the unmasked distribution is non-uniform.
    let logits: Vec<f32> = (0..ACTION_DIM_TOTAL).map(|k| (k as f32) * 0.1 - 2.0).collect();

    // Forbid every odd action; allow Hold + the evens.
    let mut mask = vec![0.0f32; ACTION_DIM_TOTAL];
    for k in 0..ACTION_DIM_TOTAL {
        if k % 2 == 1 {
            mask[k] = MASK_NEG;
        }
    }

    let probs = softmax_row(&logits, &mask, &device);

    // Sums to 1.
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "probs sum to {sum}, expected 1");

    // Forbidden (odd) actions ≈ 0; allowed (even) actions carry the mass.
    let mut allowed_mass = 0.0f32;
    for k in 0..ACTION_DIM_TOTAL {
        assert!(probs[k].is_finite(), "prob[{k}] not finite");
        if k % 2 == 1 {
            assert!(
                probs[k] < 1e-6,
                "forbidden action {k} has non-negligible prob {}",
                probs[k]
            );
        } else {
            allowed_mass += probs[k];
        }
    }
    assert!(
        (allowed_mass - 1.0).abs() < 1e-4,
        "allowed mass {allowed_mass}, expected ~1"
    );
}

#[test]
fn masked_softmax_hold_sampleable_when_all_else_masked() {
    let device = WgpuDevice::default();

    let logits: Vec<f32> = (0..ACTION_DIM_TOTAL).map(|k| (k as f32) * 0.05).collect();
    // Mask EVERYTHING (incl. Hold's column — masked_softmax must force-unmask it).
    let mask = vec![MASK_NEG; ACTION_DIM_TOTAL];

    let probs = softmax_row(&logits, &mask, &device);

    for (k, &p) in probs.iter().enumerate() {
        assert!(p.is_finite(), "prob[{k}] not finite on fully-masked row");
    }
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "probs sum to {sum}, expected 1");

    // Hold must carry essentially all the probability mass (it is the only
    // un-masked action), so it is always sampleable.
    assert!(
        probs[HOLD_ACTION] > 0.99,
        "Hold prob {} should dominate when all else is masked",
        probs[HOLD_ACTION]
    );
}

#[test]
fn masked_softmax_mostly_masked_row_finite() {
    let device = WgpuDevice::default();

    let logits = vec![3.0f32; ACTION_DIM_TOTAL]; // large equal logits stress exp()
    // Allow only Hold and one other action; forbid the rest.
    let mut mask = vec![MASK_NEG; ACTION_DIM_TOTAL];
    mask[HOLD_ACTION] = 0.0;
    let other = 5usize;
    mask[other] = 0.0;

    let probs = softmax_row(&logits, &mask, &device);

    for (k, &p) in probs.iter().enumerate() {
        assert!(p.is_finite(), "prob[{k}] not finite on mostly-masked row");
        assert!((0.0..=1.0 + 1e-6).contains(&p), "prob[{k}]={p} out of range");
    }
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "probs sum to {sum}, expected 1");
    // The two allowed equal-logit actions split the mass ~evenly.
    assert!(
        (probs[HOLD_ACTION] - probs[other]).abs() < 1e-3,
        "equal-logit allowed actions should share mass: hold={} other={}",
        probs[HOLD_ACTION],
        probs[other]
    );
}
