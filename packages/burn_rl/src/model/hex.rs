//! HexRingGather: a fixed `ACTION_CELL_INDEXES_LEN` (37) gather over the token
//! dimension. Selects the centre + surrounding hex rings from the 121-cell board
//! token sequence. The index table is a constant (non-learnable, backend-free)
//! buffer, marked `#[module(skip)]` so the `Module` machinery ignores it; the
//! index tensor is materialised on the target device at forward time.

use crate::constants::{ACTION_CELL_INDEXES, ACTION_CELL_INDEXES_LEN, BOARD_CELLS};
use burn::module::Module;
use burn::tensor::{backend::Backend, Int, Tensor, TensorData};

#[derive(Module, Debug)]
pub struct HexRingGather<B: Backend> {
    /// The 37 token indices to gather. Backend-free so it round-trips cleanly
    /// through `Module` / `AutodiffModule`.
    #[module(skip)]
    pub indices: Vec<i32>,
    pub _backend: core::marker::PhantomData<B>,
}

impl<B: Backend> HexRingGather<B> {
    pub fn new(_device: &B::Device) -> Self {
        assert_eq!(
            ACTION_CELL_INDEXES.len(),
            ACTION_CELL_INDEXES_LEN,
            "ACTION_CELL_INDEXES must have length ACTION_CELL_INDEXES_LEN"
        );
        Self {
            indices: ACTION_CELL_INDEXES.to_vec(),
            _backend: core::marker::PhantomData,
        }
    }

    /// `tokens`: `[batch, 121, dim]` → gathered `[batch, 37, dim]`.
    ///
    /// Selects `ACTION_CELL_INDEXES` along the token axis (axis 1) via
    /// `Tensor::select`, which takes a **1-D** index of length `n` (37) and
    /// resizes only that axis. The index is therefore `O(n)` to build — NOT the
    /// old `gather` path that materialised a dense `[batch, n, dim]` index tensor
    /// on the CPU every forward (≈ `batch·37·dim` i32, e.g. ~0.95M for batch 100,
    /// dim 256), which dominated batched-inference wall-clock. The table is a
    /// non-learnable constant; only the `[n]` index tensor is built per call.
    pub fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch, cells, _dim] = tokens.dims();
        debug_assert_eq!(
            cells, BOARD_CELLS,
            "HexRingGather expects {BOARD_CELLS} board tokens, got {cells}"
        );
        let device = tokens.device();

        // 1-D index `[n]`; `select` broadcasts it over batch and feature axes
        // internally. out[b, i, d] = tokens[b, indices[i], d].
        let index_data = TensorData::new(self.indices.clone(), [self.indices.len()]);
        let indices = Tensor::<B, 1, Int>::from_data(index_data, &device);
        tokens.select(1, indices)
    }
}
