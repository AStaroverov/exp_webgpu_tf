//! Grid2D sin/cos positional encoding `[rows, cols, dim]` scaled by `0.1`.
//!
//! NOT USED BY v4. The authoritative spec confirms the v4 Perceiver network uses
//! ONLY `HexRingGatherLayer` for geometry — it does NOT use `Grid2D` positional
//! encoding (nor `HexNeighborGatherLayer`, nor a noise gate). This module is
//! provided purely for completeness / parity with the TS reference and possible
//! future variants; nothing in `model/mod.rs` calls it.
//!
//! The encoding below is a standard transformer-style 2D sin/cos grid: the `dim`
//! channels are split in half, one half encoding the row index and the other the
//! column index, each with the usual interleaved sin/cos frequency schedule, and
//! the whole tensor scaled by `scale` (the spec's `0.1`).

use burn::tensor::{backend::Backend, Tensor, TensorData};

/// Build a `[rows, cols, dim]` Grid2D sin/cos positional-encoding tensor.
///
/// `dim` is split evenly: the first `dim/2` channels encode the row position and
/// the remaining channels encode the column position. Within each half, frequency
/// `i` (over `dim/4` frequency bands) uses `freq = 1 / 10000^(2i / (dim/2))`, and
/// the band's two channels are `sin(pos * freq)` and `cos(pos * freq)`. The full
/// tensor is multiplied by `scale`.
///
/// Unused in v4 (see module docs); kept for completeness.
pub fn build_grid2d_encoding<B: Backend>(
    rows: usize,
    cols: usize,
    dim: usize,
    scale: f32,
    device: &B::Device,
) -> Tensor<B, 3> {
    // Build the encoding on the CPU as a flat buffer, then load it as a constant
    // tensor (this is static geometry, computed once at construction time).
    let mut data = vec![0.0f32; rows * cols * dim];

    // Half the channels for rows, half for cols.
    let half = dim / 2;
    // Number of (sin, cos) frequency bands per half.
    let bands = half / 2;

    for r in 0..rows {
        for c in 0..cols {
            let base = (r * cols + c) * dim;
            for band in 0..bands {
                // freq = 1 / 10000^(2*band / half)
                let exponent = (2 * band) as f32 / half as f32;
                let freq = 1.0f32 / 10_000.0f32.powf(exponent);

                // Row half: channels [2*band, 2*band+1].
                let row_angle = r as f32 * freq;
                data[base + 2 * band] = (row_angle).sin() * scale;
                data[base + 2 * band + 1] = (row_angle).cos() * scale;

                // Col half: channels [half + 2*band, half + 2*band + 1].
                let col_angle = c as f32 * freq;
                data[base + half + 2 * band] = (col_angle).sin() * scale;
                data[base + half + 2 * band + 1] = (col_angle).cos() * scale;
            }
        }
    }

    let tensor_data = TensorData::new(data, [rows, cols, dim]);
    Tensor::<B, 3>::from_data(tensor_data, device)
}
