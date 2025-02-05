import { mat4 } from 'gl-matrix';

export const projectionMatrix = mat4.create();

export function createResizeSystem(canvas: HTMLCanvasElement) {
    let prevWidth = canvas.width;
    let prevHeight = canvas.height;

    mat4.ortho(projectionMatrix, 0, canvas.width, 0, canvas.height, -1, 1);
    
    return function resizeSystem() {
        if (prevWidth === canvas.width && prevHeight === canvas.height) return;
        mat4.ortho(projectionMatrix, 0, canvas.width, 0, canvas.height, -1, 1);
        prevWidth = canvas.width;
        prevHeight = canvas.height;
    };
}
