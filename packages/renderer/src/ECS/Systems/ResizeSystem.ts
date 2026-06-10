import { mat4 } from "gl-matrix";

export const projectionMatrix = mat4.create();

// Camera position in world coordinates
export const cameraPosition = { x: 0, y: 0 };

// Camera zoom: >1 zooms in, <1 zooms out (shows more of the world).
export const cameraZoom = { value: 1 };

export function setCameraPosition(x: number, y: number) {
  cameraPosition.x = x;
  cameraPosition.y = y;
}

export function setCameraZoom(zoom: number) {
  cameraZoom.value = Math.max(0.01, zoom);
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
  let prevCameraZoom = cameraZoom.value;

  canvas.width = width * pixelRatio;
  canvas.height = height * pixelRatio;
  updateProjectionMatrix(width, height);

  function updateProjectionMatrix(w: number, h: number) {
    // Center the camera: camera position becomes the center of the screen.
    // Zoom scales how many world units fit on screen (half-extents / zoom).
    const halfW = w / 2 / cameraZoom.value;
    const halfH = h / 2 / cameraZoom.value;
    const left = cameraPosition.x - halfW;
    const right = cameraPosition.x + halfW;
    const bottom = cameraPosition.y - halfH;
    const top = cameraPosition.y + halfH;
    mat4.ortho(projectionMatrix, left, right, bottom, top, -1, 1);
  }

  return function resizeSystem() {
    width = canvas.offsetWidth;
    height = canvas.offsetHeight;
    pixelRatio = getPixelRatio();

    const sizeChanged =
      prevWidth !== width || prevHeight !== height || pixelRatio !== prevPixelRatio;
    const cameraChanged =
      prevCameraX !== cameraPosition.x ||
      prevCameraY !== cameraPosition.y ||
      prevCameraZoom !== cameraZoom.value;

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
    prevCameraZoom = cameraZoom.value;
  };
}
