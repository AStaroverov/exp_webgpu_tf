import { RockPartData } from "./RockParts";
import { fbm } from "../../../../../../../lib/fbm.ts";

export type RockGridOptions = {
    cols: number;
    rows: number;
    cellSize: number;
    partSize: number;
    noiseScale?: number;
    noiseOctaves?: number;
    emptyThreshold?: number;
    seed?: number;
};

export function generateGridRockShape(options: RockGridOptions): RockPartData[] {
    const {
        cols,
        rows,
        cellSize,
        partSize,
        noiseScale = 0.1,
        noiseOctaves = 3,
        emptyThreshold = 0.2,
        seed = Math.floor(Math.random() * 1000000),
    } = options;
    
    const parts: RockPartData[] = [];
    
    // Center offset
    const offsetX = (cols * cellSize) / 2;
    const offsetY = (rows * cellSize) / 2;
    
    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            // Grid position (centered)
            const gridX = col * cellSize - offsetX + cellSize / 2;
            const gridY = row * cellSize - offsetY + cellSize / 2;
            
            // Sample noise for this cell
            const density = fbm(
                col * noiseScale + seed * 0.001,
                row * noiseScale,
                seed,
                noiseOctaves,
            );
            
            // Skip if below threshold (empty cell)
            if (density < emptyThreshold) continue;
            
            // Size based on density and partSize
            const normalizedDensity = (density - emptyThreshold) / (1 - emptyThreshold);
            const size = partSize * (1 + normalizedDensity) ** 2;
            
            // Random aspect ratio
            const aspectNoise = fbm(col * noiseScale + 100, row * noiseScale + 100, 2);
            const aspect = 0.6 + aspectNoise * 0.8; // 0.6 to 1.4
            
            const w = size * aspect;
            const h = size / aspect;
            
            // Small random position offset (keeps grid tight but natural)
            const posOffsetX = (Math.random() - 0.5) * cellSize * 0.2;
            const posOffsetY = (Math.random() - 0.5) * cellSize * 0.2;
            
            // Random rotation
            const rotation = Math.random() * Math.PI * 2;
            
            parts.push([
                gridX + posOffsetX,
                gridY + posOffsetY,
                w,
                h,
                rotation,
            ]);
        }
    }
    
    return parts;
}

