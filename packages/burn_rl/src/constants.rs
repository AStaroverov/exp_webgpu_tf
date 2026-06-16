//! Compile-time constants for the v4 Perceiver actor-critic and the PPO/Retrace
//! training loop. These mirror the TypeScript reference implementation (`v4.ts`)
//! and the action-space layout; they are the single source of truth on the Rust
//! side. See the migration spec for provenance.

// ---- Board / observation geometry ----
pub const BOARD_ROWS: usize = 11;
pub const BOARD_COLS: usize = 11;
pub const BOARD_CELLS: usize = 121; // BOARD_ROWS * BOARD_COLS
pub const BOARD_CHANNELS: usize = 19;

// ---- Action space layout ----
pub const MOVE_DIR_COUNT: usize = 6;
pub const FIRE_TARGET_COUNT: usize = 36;
pub const ACTION_DIM_TOTAL: usize = 43; // hold(1) + move(6) + fire(36)
pub const HOLD_ACTION: usize = 0;
pub const MOVE_ACTION_OFFSET: usize = 1;
pub const FIRE_ACTION_OFFSET: usize = 7;

/// Number of hex cells gathered by the HexRingGather layer (centre + rings).
pub const ACTION_CELL_INDEXES_LEN: usize = 37;

/// Additive logit mask value for forbidden actions (≈ -inf before softmax).
pub const MASK_NEG: f32 = -1e9;

// ---- Perceiver policy branch ----
pub const PERCEIVER_POLICY_DIM: usize = 256;
pub const PERCEIVER_POLICY_HEADS: usize = 8;
pub const PERCEIVER_POLICY_DEPTH: usize = 4;

// ---- Perceiver value branch ----
pub const PERCEIVER_VALUE_DIM: usize = 128;
pub const PERCEIVER_VALUE_HEADS: usize = 4;
pub const PERCEIVER_VALUE_DEPTH: usize = 2;

// ---- Init / numerics ----
pub const LOGIT_INIT_GAIN: f64 = 0.2;
pub const ADAMW_EPS: f64 = 1e-5;

/// Fire-target ring radius (rings 1..=FIRE_RING_RADIUS around the tank → 36
/// fire targets). Mirrors `FIRE_RING_RADIUS` in `state/hexNeighbors.ts`.
pub const FIRE_RING_RADIUS: i32 = 3;

/// Axial (dq, dr) delta per pointy-top hex direction, in `POINTY_DIRECTIONS`
/// order (E, NE, NW, W, SW, SE). Derived from `honeycomb-grid` on the TS side
/// (`AXIAL_DELTAS` in `state/hexNeighbors.ts`); reproduced here as the computed
/// constant values so the gather table matches the game's hex topology.
pub const AXIAL_DELTAS: [(i32, i32); MOVE_DIR_COUNT] =
    [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];

/// The 37 board-cell indices gathered by `HexRingGather`: the window centre
/// (self) followed by each fire-target ring cell, in `FIRE_TARGET_OFFSETS`
/// order. Computed by `build_action_cell_indexes()` to mirror the TS
/// construction in `state/hexNeighbors.ts` (so it stays correct if dims change),
/// then validated (length, in-range, distinct) by `assert_action_cell_indexes`.
pub const ACTION_CELL_INDEXES: [i32; ACTION_CELL_INDEXES_LEN] = build_action_cell_indexes();

/// Const-fn reproduction of the TS `ACTION_CELL_INDEXES` construction
/// (`state/hexNeighbors.ts`): centre = VIEW_RADIUS*BOARD_COLS + VIEW_RADIUS,
/// then for each fire-target offset (rings 1..=FIRE_RING_RADIUS, walked ring by
/// ring exactly as `FIRE_TARGET_OFFSETS`) push (VIEW_RADIUS+dr)*BOARD_COLS +
/// (VIEW_RADIUS+dq).
const fn build_action_cell_indexes() -> [i32; ACTION_CELL_INDEXES_LEN] {
    const VIEW_RADIUS: i32 = (BOARD_COLS as i32 - 1) / 2;
    const COLS: i32 = BOARD_COLS as i32;
    let mut out = [0i32; ACTION_CELL_INDEXES_LEN];
    // Centre / self cell (window centre).
    out[0] = VIEW_RADIUS * COLS + VIEW_RADIUS;
    let mut n = 1usize;
    let mut k = 1i32;
    while k <= FIRE_RING_RADIUS {
        let mut side = 0usize;
        while side < MOVE_DIR_COUNT {
            let (cq, cr) = AXIAL_DELTAS[side];
            // Stepping from corner i toward corner i+1 follows direction i+2.
            let (sq, sr) = AXIAL_DELTAS[(side + 2) % MOVE_DIR_COUNT];
            let mut step = 0i32;
            while step < k {
                let dq = k * cq + step * sq;
                let dr = k * cr + step * sr;
                let col = VIEW_RADIUS + dq;
                let row = VIEW_RADIUS + dr;
                out[n] = row * COLS + col;
                n += 1;
                step += 1;
            }
            side += 1;
        }
        k += 1;
    }
    out
}

/// Debug assertion that the gather table is well-formed: every index is in
/// `[0, BOARD_CELLS)` and all 37 are DISTINCT. A length check alone did not
/// catch the all-zeros placeholder bug, so the distinctness check is the gate.
pub fn assert_action_cell_indexes() {
    debug_assert_eq!(
        ACTION_CELL_INDEXES.len(),
        ACTION_CELL_INDEXES_LEN,
        "ACTION_CELL_INDEXES length mismatch"
    );
    let mut seen = [false; BOARD_CELLS];
    for &idx in ACTION_CELL_INDEXES.iter() {
        debug_assert!(
            idx >= 0 && (idx as usize) < BOARD_CELLS,
            "ACTION_CELL_INDEXES entry {idx} out of range [0, {BOARD_CELLS})"
        );
        debug_assert!(
            !seen[idx as usize],
            "ACTION_CELL_INDEXES contains duplicate cell index {idx}"
        );
        seen[idx as usize] = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_cell_indexes_match_ts() {
        // Computed from state/hexNeighbors.ts (node + honeycomb-grid).
        let expected: [i32; ACTION_CELL_INDEXES_LEN] = [
            60, 61, 50, 49, 59, 70, 71, 62, 51, 40, 39, 38, 48, 58, 69, 80, 81, 82, 72, 63, 52, 41,
            30, 29, 28, 27, 37, 47, 57, 68, 79, 90, 91, 92, 93, 83, 73,
        ];
        assert_eq!(ACTION_CELL_INDEXES, expected);
        assert_action_cell_indexes();
    }
}
