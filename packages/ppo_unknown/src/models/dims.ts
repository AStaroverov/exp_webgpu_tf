/**
 * Model dims for ppo_unknown — the board observation shape + the action head
 * layout. Observation dims come straight from the board store; action dims from
 * consts. Keeping them here lets the network code stay parametric.
 */

export { BOARD_ROWS, BOARD_COLS, BOARD_CHANNELS, BOARD_CELLS, BOARD_SIZE, BoardChannel } from '../state/board.ts';
export { ACTION_HEAD_DIMS, MOVE_DIR_COUNT, FIRE_DIR_COUNT } from '../consts.ts';
