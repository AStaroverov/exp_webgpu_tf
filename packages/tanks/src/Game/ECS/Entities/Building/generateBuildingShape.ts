import { BuildingPartData } from "./BuildingParts.ts";
import { fbm } from "../../../../../../../lib/fbm.ts";

export type BuildingGridOptions = {
    /** Number of columns (width in cells) */
    cols: number;
    /** Number of rows (height in cells) */
    rows: number;
    /** Size of each cell */
    cellSize: number;
    /** Wall thickness */
    wallThickness: number;
    /** Noise scale for damage/destruction */
    noiseScale?: number;
    /** Noise octaves */
    noiseOctaves?: number;
    /** Threshold for wall destruction (higher = more destruction) */
    destructionThreshold?: number;
    /** Chance for interior walls */
    interiorWallChance?: number;
    /** Seed for randomization */
    seed?: number;
};

export type BuildingPartType = 'wall' | 'floor';

/**
 * Generate a ruined building shape using a grid-based approach.
 * Creates walls along cell edges with random destruction based on noise.
 */
export function generateBuildingShape(options: BuildingGridOptions): BuildingPartData[] {
    const {
        cols,
        rows,
        cellSize,
        wallThickness,
        noiseScale = 0.15,
        noiseOctaves = 2,
        destructionThreshold = 0.3,
        interiorWallChance = 0.5,
        seed = Math.floor(Math.random() * 1000000),
    } = options;

    const parts: BuildingPartData[] = [];

    // Center offset
    const offsetX = (cols * cellSize) / 2;
    const offsetY = (rows * cellSize) / 2;

    // Track which walls exist for interior generation
    const horizontalWalls: boolean[][] = [];
    const verticalWalls: boolean[][] = [];

    // Initialize wall arrays
    for (let row = 0; row <= rows; row++) {
        horizontalWalls[row] = [];
        for (let col = 0; col < cols; col++) {
            horizontalWalls[row][col] = false;
        }
    }
    for (let row = 0; row < rows; row++) {
        verticalWalls[row] = [];
        for (let col = 0; col <= cols; col++) {
            verticalWalls[row][col] = false;
        }
    }

    // Generate exterior walls (perimeter)
    // Top and bottom walls
    for (let col = 0; col < cols; col++) {
        const topDamage = fbm(col * noiseScale, 0, seed, noiseOctaves);
        const bottomDamage = fbm(col * noiseScale, rows * noiseScale, seed + 100, noiseOctaves);

        // Top wall
        if (topDamage > destructionThreshold * 0.5) {
            const x = col * cellSize - offsetX + cellSize / 2;
            const y = -offsetY;
            const damageMultiplier = Math.min(1, (topDamage - destructionThreshold * 0.5) / 0.5);
            const width = cellSize * (0.5 + damageMultiplier * 0.5);
            const height = wallThickness * (0.6 + damageMultiplier * 0.4);

            parts.push({
                x: x + (Math.random() - 0.5) * width * 0.15,
                y: y + (Math.random() - 0.5) * height * 0.15,
                width,
                height,
                rotation: (Math.random() - 0.5) * 0.12,
                type: 'wall',
            });
            horizontalWalls[0][col] = true;
        }

        // Bottom wall
        if (bottomDamage > destructionThreshold * 0.5) {
            const x = col * cellSize - offsetX + cellSize / 2;
            const y = rows * cellSize - offsetY;
            const damageMultiplier = Math.min(1, (bottomDamage - destructionThreshold * 0.5) / 0.5);
            const width = cellSize * (0.5 + damageMultiplier * 0.5);
            const height = wallThickness * (0.6 + damageMultiplier * 0.4);

            parts.push({
                x: x + (Math.random() - 0.5) * width * 0.15,
                y: y + (Math.random() - 0.5) * height * 0.15,
                width,
                height,
                rotation: (Math.random() - 0.5) * 0.12,
                type: 'wall',
            });
            horizontalWalls[rows][col] = true;
        }
    }

    // Left and right walls
    for (let row = 0; row < rows; row++) {
        const leftDamage = fbm(0, row * noiseScale, seed + 200, noiseOctaves);
        const rightDamage = fbm(cols * noiseScale, row * noiseScale, seed + 300, noiseOctaves);

        // Left wall
        if (leftDamage > destructionThreshold * 0.5) {
            const x = -offsetX;
            const y = row * cellSize - offsetY + cellSize / 2;
            const damageMultiplier = Math.min(1, (leftDamage - destructionThreshold * 0.5) / 0.5);
            const width = wallThickness * (0.6 + damageMultiplier * 0.4);
            const height = cellSize * (0.5 + damageMultiplier * 0.5);

            parts.push({
                x: x + (Math.random() - 0.5) * width * 0.15,
                y: y + (Math.random() - 0.5) * height * 0.15,
                width,
                height,
                rotation: (Math.random() - 0.5) * 0.12,
                type: 'wall',
            });
            verticalWalls[row][0] = true;
        }

        // Right wall
        if (rightDamage > destructionThreshold * 0.5) {
            const x = cols * cellSize - offsetX;
            const y = row * cellSize - offsetY + cellSize / 2;
            const damageMultiplier = Math.min(1, (rightDamage - destructionThreshold * 0.5) / 0.5);
            const width = wallThickness * (0.6 + damageMultiplier * 0.4);
            const height = cellSize * (0.5 + damageMultiplier * 0.5);

            parts.push({
                x: x + (Math.random() - 0.5) * width * 0.15,
                y: y + (Math.random() - 0.5) * height * 0.15,
                width,
                height,
                rotation: (Math.random() - 0.5) * 0.12,
                type: 'wall',
            });
            verticalWalls[row][cols] = true;
        }
    }

    // Generate interior walls
    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            const interiorNoise = fbm(
                col * noiseScale * 2 + 50,
                row * noiseScale * 2 + 50,
                seed + 400,
                noiseOctaves,
            );

            // Interior horizontal wall (below this cell)
            if (row < rows - 1) {
                const hWallNoise = fbm(col * noiseScale, (row + 1) * noiseScale, seed + 500, noiseOctaves);
                if (interiorNoise > (1 - interiorWallChance) && hWallNoise > destructionThreshold) {
                    const x = col * cellSize - offsetX + cellSize / 2;
                    const y = (row + 1) * cellSize - offsetY;
                    const damageMultiplier = Math.min(1, (hWallNoise - destructionThreshold) / 0.5);
                    const width = cellSize * (0.4 + damageMultiplier * 0.5);
                    const height = wallThickness * (0.5 + damageMultiplier * 0.4);

                    parts.push({
                        x: x + (Math.random() - 0.5) * width * 0.22,
                        y: y + (Math.random() - 0.5) * height * 0.22,
                        width,
                        height,
                        rotation: (Math.random() - 0.5) * 0.18,
                        type: 'wall',
                    });
                }
            }

            // Interior vertical wall (to the right of this cell)
            if (col < cols - 1) {
                const vWallNoise = fbm((col + 1) * noiseScale, row * noiseScale, seed + 600, noiseOctaves);
                if (interiorNoise > (1 - interiorWallChance * 0.8) && vWallNoise > destructionThreshold) {
                    const x = (col + 1) * cellSize - offsetX;
                    const y = row * cellSize - offsetY + cellSize / 2;
                    const damageMultiplier = Math.min(1, (vWallNoise - destructionThreshold) / 0.5);
                    const width = wallThickness * (0.5 + damageMultiplier * 0.4);
                    const height = cellSize * (0.4 + damageMultiplier * 0.5);

                    parts.push({
                        x: x + (Math.random() - 0.5) * width * 0.22,
                        y: y + (Math.random() - 0.5) * height * 0.22,
                        width,
                        height,
                        rotation: (Math.random() - 0.5) * 0.18,
                        type: 'wall',
                    });
                }
            }
        }
    }

    // Generate floor tiles in clusters/islands using noise
    // Tile size is small for detail
    const tileSize = 8;
    // Noise threshold - higher means fewer, more clustered tiles
    const tileThreshold = 0.6 + Math.random() * 0.2;
    
    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            // Room boundaries
            const roomLeft = col * cellSize - offsetX;
            const roomTop = row * cellSize - offsetY;
            
            // Calculate how many tiles fit in this room
            const tilesPerRow = Math.ceil(cellSize / tileSize);
            const tilesPerCol = Math.ceil(cellSize / tileSize);
            
            for (let tileRow = 0; tileRow < tilesPerCol; tileRow++) {
                for (let tileCol = 0; tileCol < tilesPerRow; tileCol++) {
                    // Base tile position
                    const baseTileX = roomLeft + tileCol * tileSize + tileSize / 2;
                    const baseTileY = roomTop + tileRow * tileSize + tileSize / 2;
                    
                    // Use noise to create islands/clusters of tiles
                    const tileNoise = fbm(
                        baseTileX * 0.03 + seed * 0.001,
                        baseTileY * 0.03,
                        seed + 1000,
                        2,
                    );
                    
                    // Only spawn tile if noise is above threshold (creates clusters)
                    if (tileNoise < tileThreshold) continue;
                    
                    // Randomize position for "broken/displaced" effect
                    const displaceX = (Math.random() - 0.5) * tileSize * 0.6;
                    const displaceY = (Math.random() - 0.5) * tileSize * 0.6;
                    const tileX = baseTileX + displaceX;
                    const tileY = baseTileY + displaceY;
                    
                    // Size variation for natural look
                    const sizeVariation = 0.7 + Math.random() * 0.5;
                    
                    // Small random rotation for "displaced" effect
                    const tileRotation = (Math.random() - 0.5) * 0.22;
                    
                    parts.push({
                        x: tileX,
                        y: tileY,
                        width: tileSize * sizeVariation,
                        height: tileSize * sizeVariation * (0.8 + Math.random() * 0.4),
                        rotation: tileRotation,
                        type: 'floor',
                    });
                }
            }
        }
    }

    // Add corner pillars (more likely to survive)
    const corners = [
        { x: -offsetX, y: -offsetY },
        { x: cols * cellSize - offsetX, y: -offsetY },
        { x: -offsetX, y: rows * cellSize - offsetY },
        { x: cols * cellSize - offsetX, y: rows * cellSize - offsetY },
    ];

    corners.forEach((corner, i) => {
        const cornerNoise = fbm(corner.x * 0.01, corner.y * 0.01, seed + 800 + i, noiseOctaves);
        if (cornerNoise > destructionThreshold * 0.3) {
            const sizeMultiplier = 0.7 + cornerNoise * 0.5;
            const pillarSize = wallThickness * 1.5 * sizeMultiplier;
            parts.push({
                x: corner.x + (Math.random() - 0.5) * pillarSize * 0.15,
                y: corner.y + (Math.random() - 0.5) * pillarSize * 0.15,
                width: pillarSize,
                height: pillarSize,
                rotation: (Math.random() - 0.5) * 0.15,
                type: 'wall',
            });
        }
    });

    return parts;
}

