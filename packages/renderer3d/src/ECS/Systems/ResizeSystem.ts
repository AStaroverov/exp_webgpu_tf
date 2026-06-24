import { mat4, vec3 } from "gl-matrix";

// Orthographic tilted top-down (2.5D) camera.
//
// DEPTH CONVENTION — REVERSE-Z. The draw pipeline uses depthCompare
// "greater-equal" with depthClearValue 0 (see GPUShader.ts withDepth and
// createFrame.ts). So the clip-space depth this matrix produces must map
// NEAR -> 1 and FAR -> 0 (nearer = larger depth). We achieve it with an
// "ndc-fix" that remaps gl-matrix's OpenGL z range [-1, 1] to WebGPU REVERSED
// [1, 0] (instead of the usual [0, 1]): z' = -0.5*z + 0.5.
// The shader (sdf.shader.ts) writes frag_depth = (viewProj * hitWorld).z / .w
// and MUST share this exact convention.

// viewProj : orthographic tilted top-down, reverse-Z (NEAR=1 .. FAR=0).
export const viewProjMatrix = mat4.create();
// cameraRayDir : world-space camera forward = normalize(target - eye). Constant
// across the frame for an orthographic camera (all rays parallel).
export const cameraRayDir = vec3.create();
// sceneLightDir : fixed world-space directional light (points along travel).
export const sceneLightDir = vec3.normalize(
  vec3.create(),
  vec3.fromValues(-0.4, -0.55, -0.72),
);

// Camera position in world coordinates (the look-at target on the ground plane).
export const cameraPosition = { x: 0, y: 0 };

// Camera zoom: >1 zooms in, <1 zooms out (shows more of the world).
export const cameraZoom = { value: 1 };

// Elevation: angle above the ground plane in degrees (90 = straight down).
// ~70 matches the prototype.
export const cameraElevation = { value: 70 };

// Azimuth: orbit angle around the Z axis, in degrees.
export const cameraAzimuth = { value: 45 };

export function setCameraPosition(x: number, y: number) {
  cameraPosition.x = x;
  cameraPosition.y = y;
}

export function setCameraZoom(zoom: number) {
  cameraZoom.value = Math.max(0.01, zoom);
}

export function setCameraElevation(degrees: number) {
  cameraElevation.value = Math.max(1, Math.min(89.9, degrees));
}

export function setCameraAzimuth(degrees: number) {
  cameraAzimuth.value = degrees;
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
  let prevCameraElevation = cameraElevation.value;
  let prevCameraAzimuth = cameraAzimuth.value;

  canvas.width = width * pixelRatio;
  canvas.height = height * pixelRatio;

  // Scratch matrices/vectors reused every update (no per-frame allocation).
  const view = mat4.create();
  const proj = mat4.create();
  // Remap gl-matrix OpenGL depth [-1, 1] to WebGPU REVERSED [1, 0] (reverse-Z).
  const ndcFix = mat4.fromValues(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -0.5, 0, 0, 0, 0.5, 1);
  const eye = vec3.create();
  const dir = vec3.create();
  const target = vec3.create();

  // Initial projection: must run AFTER the scratch consts above exist
  // (updateProjectionMatrix closes over them — calling earlier hits the TDZ).
  updateProjectionMatrix(width, height);

  function updateProjectionMatrix(w: number, h: number) {
    // World half-height visible on screen, scaled by zoom; X follows aspect.
    const aspect = w / h;
    const halfH = h / 2 / cameraZoom.value;
    const halfW = halfH * aspect;

    const elevation = (cameraElevation.value * Math.PI) / 180;
    const azimuth = (cameraAzimuth.value * Math.PI) / 180;

    // Look-at target sits on the camera position; raise it slightly so the
    // visible volume is roughly centered (mirrors the prototype's target.z).
    target[0] = cameraPosition.x;
    target[1] = cameraPosition.y;
    target[2] = 0;

    // Camera sits along a tilted direction at a large fixed distance; the
    // orthographic near/far planes give the usable depth range.
    const ce = Math.cos(elevation);
    const se = Math.sin(elevation);
    const dist = 100;
    dir[0] = ce * Math.cos(azimuth);
    dir[1] = ce * Math.sin(azimuth);
    dir[2] = se;
    vec3.scaleAndAdd(eye, target, dir, dist);
    mat4.lookAt(view, eye, target, vec3.fromValues(0, 0, 1));

    mat4.ortho(proj, -halfW, halfW, -halfH, halfH, 0.1, dist + 200);
    // Reverse-Z remap, then compose viewProj = ndcFix * proj * view.
    mat4.multiply(proj, ndcFix, proj);
    mat4.multiply(viewProjMatrix, proj, view);

    // Orthographic: every camera ray is parallel. cameraRayDir = forward.
    vec3.subtract(cameraRayDir, target, eye);
    vec3.normalize(cameraRayDir, cameraRayDir);
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
      prevCameraZoom !== cameraZoom.value ||
      prevCameraElevation !== cameraElevation.value ||
      prevCameraAzimuth !== cameraAzimuth.value;

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
    prevCameraElevation = cameraElevation.value;
    prevCameraAzimuth = cameraAzimuth.value;
  };
}
