import { mat4 } from 'gl-matrix';

export const projectionMatrix = mat4.create();

export function createResizeSystem(canvas: HTMLCanvasElement, pixelRatio: number) {
    let width = canvas.width / pixelRatio;
    let height = canvas.height / pixelRatio;
    let prevWidth = canvas.width / pixelRatio;
    let prevHeight = canvas.height / pixelRatio;

    mat4.ortho(projectionMatrix, 0, width, 0, height, -1, 1);

    return function resizeSystem() {
        width = canvas.width / pixelRatio;
        height = canvas.height / pixelRatio;
        if (prevWidth === width && prevHeight === height) return;
        mat4.ortho(projectionMatrix, 0, width, 0, height, -1, 1);
        prevWidth = width;
        prevHeight = height;
    };
}
