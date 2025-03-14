import { mat4 } from 'gl-matrix';

export const projectionMatrix = mat4.create();

export function createResizeSystem(canvas: HTMLCanvasElement, getPixelRatio: () => number) {
    let width = canvas.offsetWidth;
    let height = canvas.offsetHeight;
    let prevWidth = width;
    let prevHeight = height;
    let pixelRatio = getPixelRatio();
    let prevPixelRatio = pixelRatio;

    canvas.width = width * pixelRatio;
    canvas.height = height * pixelRatio;
    mat4.ortho(projectionMatrix, 0, width, 0, height, -1, 1);

    return function resizeSystem() {
        width = canvas.offsetWidth;
        height = canvas.offsetHeight;
        pixelRatio = getPixelRatio();
        if (prevWidth === width && prevHeight === height && pixelRatio === prevPixelRatio) return;
        debugger
        canvas.width = width * pixelRatio;
        canvas.height = height * pixelRatio;
        mat4.ortho(projectionMatrix, 0, width, 0, height, -1, 1);
        prevWidth = width;
        prevHeight = height;
        prevPixelRatio = pixelRatio;
    };
}
