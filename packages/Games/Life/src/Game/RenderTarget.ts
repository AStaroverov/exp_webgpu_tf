import { BehaviorSubject, combineLatest } from 'rxjs';
import { engine$ } from './engine.js';

export const targetCanvas$ = new BehaviorSubject<HTMLCanvasElement | null>(null);

combineLatest([engine$, targetCanvas$]).subscribe(([engine, canvas]) => {
    engine?.setRenderTarget(canvas);
});

export const setRenderTarget = (canvas: HTMLCanvasElement | null) => {
    targetCanvas$.next(canvas);
};