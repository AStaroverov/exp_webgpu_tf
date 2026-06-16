//! Checkpoint save / load: bincode-serialised model weights plus JSON metadata
//! (experiment iteration, curriculum state). The curriculum itself lives in TS;
//! we only round-trip its opaque serialized state here (see migration risks).

use crate::model::ActorCriticV4;
use burn::backend::wgpu::Wgpu;
use serde::{Deserialize, Serialize};

/// Metadata stored alongside the weights.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub exp_iteration: u64,
    /// Opaque curriculum state owned by the TS side; round-tripped as JSON.
    pub curriculum_state: serde_json::Value,
}

/// Serialize the model weights (bincode) + metadata (JSON) into a byte blob.
pub fn save_checkpoint<B>(_model: &ActorCriticV4<B>, _metadata: &CheckpointMetadata) -> Vec<u8>
where
    B: burn::tensor::backend::Backend,
{
    unimplemented!("save_checkpoint: bincode weights + JSON metadata")
}

/// Load a checkpoint blob into a `Wgpu` model + metadata.
pub fn load_checkpoint(_bytes: &[u8]) -> (ActorCriticV4<Wgpu>, CheckpointMetadata) {
    unimplemented!("load_checkpoint: bincode weights + JSON metadata")
}
