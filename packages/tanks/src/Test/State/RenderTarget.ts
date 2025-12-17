import { BehaviorSubject, combineLatest } from 'rxjs';
import { engine$ } from './engine.ts';

const targetCanvas$ = new BehaviorSubject<HTMLCanvasElement | null>(null);

combineLatest([engine$, targetCanvas$]).subscribe(([engine, canvas]) => {
    engine?.setRenderTarget(canvas);
    engine?.enableSound();
});

export const setRenderTarget = (canvas: HTMLCanvasElement | null) => {
    targetCanvas$.next(canvas);
};

