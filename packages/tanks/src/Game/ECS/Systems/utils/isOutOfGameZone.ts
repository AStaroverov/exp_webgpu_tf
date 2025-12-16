import { GameDI } from '../../../DI/GameDI.ts';
import { GameMap } from '../../Entities/GameMap.ts';

/**
 * Check if position is out of the game zone.
 * Zone is always centered around GameMap offset with size (width × height).
 * In bounded mode offset = (width/2, height/2), in infinite mode offset follows player.
 */
export function isOutOfGameZone(x: number, y: number, shift: number) {
    const { width, height } = GameDI;
    const { offsetX, offsetY } = GameMap;
    
    const minX = offsetX - width / 2 - shift;
    const maxX = offsetX + width / 2 + shift;
    const minY = offsetY - height / 2 - shift;
    const maxY = offsetY + height / 2 + shift;
    
    return x < minX || x > maxX || y < minY || y > maxY;
}