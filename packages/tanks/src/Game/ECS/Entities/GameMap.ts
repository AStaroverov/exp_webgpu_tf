/**
 * GameMap stores the world offset for infinite map support.
 * 
 * In training mode: offset stays at (0, 0) - bounded map
 * In bullet hell mode: offset follows the player - infinite map illusion
 * 
 * All game entities use world coordinates.
 * The offset is subtracted during rendering to center the camera on the player.
 * The offset is also used when collecting input tensors for ML models,
 * so the model sees coordinates relative to the map origin, not the player.
 */
export const GameMap = {
    // World offset - represents how much the "virtual origin" has shifted
    offsetX: 0,
    offsetY: 0,

    /**
     * Set the map offset (used in bullet hell to follow player)
     */
    setOffset(x: number, y: number) {
        GameMap.offsetX = x;
        GameMap.offsetY = y;
    },

    /**
     * Get offset as tuple
     */
    getOffset(): [number, number] {
        return [GameMap.offsetX, GameMap.offsetY];
    },

    /**
     * Reset offset to origin (used when destroying/resetting game)
     */
    reset() {
        GameMap.offsetX = 0;
        GameMap.offsetY = 0;
    },

    /**
     * Convert world coordinates to view coordinates (for rendering)
     * viewCoord = worldCoord - offset
     */
    worldToView(worldX: number, worldY: number): [number, number] {
        return [worldX - GameMap.offsetX, worldY - GameMap.offsetY];
    },

    /**
     * Convert view coordinates to world coordinates
     * worldCoord = viewCoord + offset
     */
    viewToWorld(viewX: number, viewY: number): [number, number] {
        return [viewX + GameMap.offsetX, viewY + GameMap.offsetY];
    },
};
