import { mat4 } from 'gl-matrix';

export const projectionMatrix = mat4.create();

// Camera position in world coordinates
export const cameraPosition = { x: 0, y: 0 };

export function setCameraPosition(x: number, y: number) {
    cameraPosition.x = x;
    cameraPosition.y = y;
}

export function createResizeSystem(canvas: HTMLCanvasElement, getPixelRatio: () => number) {
    let width = canvas.offsetWidth;
    let height = canvas.offsetHeight;
    let prevWidth = width;
    let prevHeight = height;
    let pixelRatio = getPixelRatio();
    let prevPixelRatio = pixelRatio;
    let prevCameraX = cameraPosition.x;
    let prevCameraY = cameraPosition.y;

    canvas.width = width * pixelRatio;
    canvas.height = height * pixelRatio;
    updateProjectionMatrix(width, height);

    function updateProjectionMatrix(w: number, h: number) {
        // Center the camera: camera position becomes the center of the screen
        const left = cameraPosition.x - w / 2;
        const right = cameraPosition.x + w / 2;
        const bottom = cameraPosition.y - h / 2;
        const top = cameraPosition.y + h / 2;
        mat4.ortho(projectionMatrix, left, right, bottom, top, -1, 1);
    }

    return function resizeSystem() {
        width = canvas.offsetWidth;
        height = canvas.offsetHeight;
        pixelRatio = getPixelRatio();
        
        const sizeChanged = prevWidth !== width || prevHeight !== height || pixelRatio !== prevPixelRatio;
        const cameraChanged = prevCameraX !== cameraPosition.x || prevCameraY !== cameraPosition.y;
        
        if (!sizeChanged && !cameraChanged) return;
        
        if (sizeChanged) {
            canvas.width = width * pixelRatio;
            canvas.height = height * pixelRatio;
        }
        
        updateProjectionMatrix(width, height);
        
        prevWidth = width;
        prevHeight = height;
        prevPixelRatio = pixelRatio;
        prevCameraX = cameraPosition.x;
        prevCameraY = cameraPosition.y;
    };
}
