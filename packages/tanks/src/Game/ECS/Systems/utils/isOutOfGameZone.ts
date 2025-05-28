import { GameDI } from '../../../DI/GameDI.ts';

export function isOutOfGameZone(x: number, y: number, shift: number) {
    const { width, height } = GameDI;
    return x < -shift || x > width + shift || y < -shift || y > height + shift;
}