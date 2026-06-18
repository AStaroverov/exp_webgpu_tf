import { mat4, vec3 } from "gl-matrix";

// Combined view-projection (reverse-Z). The name `projectionMatrix` is kept so
// the existing uploaders (shape / rope / VFX / grid) need no change: the shader's
// `uProjection * model` now reads `VP * model`, which is correct.
export const projectionMatrix = mat4.create(); // = VP (reverse-Z)
// uInvViewProj — inverse of the reverse-Z VP, for ray/world reconstruction (Phase 1+).
export const invViewProjection = mat4.create();
// uCameraPos — eye position in world space (Phase 1 SDF-impostor ray origin).
export const cameraPosition = vec3.fromValues(0, 0, 0);

// --- Reverse-Z remap: z' = 1 - z, leaves x,y,w untouched -------------------
// gl-matrix 3.4.3 `mat4.perspective` already targets WebGPU [0,1] depth (the ZO
// form). Post-multiplying this remap maps NDC z=0->1 and z=1->0, so near->1 /
// far->0 — matching depthClearValue:0 + depthCompare:'greater-equal'.
// Column-major: multiplying clip (x,y,z,w) gives x'=x, y'=y, z'=w-z, w'=w =>
// after /w : ndcZ' = 1 - ndcZ.
// DEPTH CONVENTION (pin): gl-matrix 3.x perspective == WebGPU [0,1]. If bumped to
// gl-matrix 4 (perspectiveNO/ZO split) this remap stays valid but a plain
// `perspective` could regress to GL [-1,1]; keep the ZO form.
const REVERSE_Z = mat4.fromValues(
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, -1, 0,
  0, 0, 1, 1,
);

const FOVY = (60 * Math.PI) / 180; // vertical field of view
const NEAR = 0.1;
const FAR = 20000;
const BASE_DISTANCE = 600;
const PITCH_LIMIT = Math.PI / 2 - 0.01;

export interface OrbitCamera {
  yaw: number; // radians, around world +Y
  pitch: number; // radians, clamped away from poles
  distance: number; // eye distance from target
  target: vec3; // world point looked at (follow this)
}

export const orbit: OrbitCamera = {
  yaw: 0,
  pitch: 1.1, // positive pitch = eye above the ground (sin(pitch)>0), looking down
  distance: BASE_DISTANCE,
  target: vec3.fromValues(0, 0, 0),
};

// spherical -> cartesian eye offset (world +Y up). Writes into out, no alloc.
function computeEye(out: vec3): vec3 {
  const cp = Math.cos(orbit.pitch);
  out[0] = orbit.target[0] + orbit.distance * cp * Math.sin(orbit.yaw);
  out[1] = orbit.target[1] + orbit.distance * Math.sin(orbit.pitch);
  out[2] = orbit.target[2] + orbit.distance * cp * Math.cos(orbit.yaw);
  return out;
}

// Follow API (replaces setCameraPosition): move the orbit target. Gameplay is
// still XY-planar (z=0), so map gameplay (x,y) -> target (x, y, 0).
export function setCameraTarget(x: number, y: number, z = 0) {
  orbit.target[0] = x;
  orbit.target[1] = y;
  orbit.target[2] = z;
}

// Back-compat alias: old callers said setCameraPosition(x, y).
export const setCameraPosition = setCameraTarget;

// Replaces setCameraZoom: zoom == dolly the orbit distance. Old zoom>1 == zoom
// in == smaller distance, so invert.
export function setCameraZoom(zoom: number) {
  orbit.distance = BASE_DISTANCE / Math.max(0.01, zoom);
}

export function orbitBy(dYaw: number, dPitch: number) {
  orbit.yaw += dYaw;
  orbit.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, orbit.pitch + dPitch));
}

// scratch (module-level, never realloc in the hot path)
const _proj = mat4.create();
const _view = mat4.create();
const _eye = vec3.create();
const _up = vec3.fromValues(0, 1, 0); // world +Y is up (right-handed world)

function updateViewProjection(w: number, h: number) {
  const aspect = w / h;

  mat4.perspective(_proj, FOVY, aspect, NEAR, FAR);
  // Bake reverse-Z: Pz = REVERSE_Z * P
  mat4.multiply(_proj, REVERSE_Z, _proj);

  computeEye(_eye);
  mat4.lookAt(_view, _eye, orbit.target, _up);

  // VP = Pz * V
  mat4.multiply(projectionMatrix, _proj, _view);

  mat4.invert(invViewProjection, projectionMatrix);
  vec3.copy(cameraPosition, _eye);
}

export function createResizeSystem(canvas: HTMLCanvasElement, getPixelRatio: () => number) {
  let width = canvas.offsetWidth;
  let height = canvas.offsetHeight;
  let prevWidth = width;
  let prevHeight = height;
  let pixelRatio = getPixelRatio();
  let prevPixelRatio = pixelRatio;

  // Snapshot the orbit state for the dirty check.
  let prevYaw = orbit.yaw;
  let prevPitch = orbit.pitch;
  let prevDistance = orbit.distance;
  let prevTargetX = orbit.target[0];
  let prevTargetY = orbit.target[1];
  let prevTargetZ = orbit.target[2];

  canvas.width = width * pixelRatio;
  canvas.height = height * pixelRatio;
  updateViewProjection(width, height);

  return function resizeSystem() {
    width = canvas.offsetWidth;
    height = canvas.offsetHeight;
    pixelRatio = getPixelRatio();

    const sizeChanged =
      prevWidth !== width || prevHeight !== height || pixelRatio !== prevPixelRatio;
    const cameraChanged =
      prevYaw !== orbit.yaw ||
      prevPitch !== orbit.pitch ||
      prevDistance !== orbit.distance ||
      prevTargetX !== orbit.target[0] ||
      prevTargetY !== orbit.target[1] ||
      prevTargetZ !== orbit.target[2];

    if (!sizeChanged && !cameraChanged) return;

    if (sizeChanged) {
      canvas.width = width * pixelRatio;
      canvas.height = height * pixelRatio;
    }

    updateViewProjection(width, height);

    prevWidth = width;
    prevHeight = height;
    prevPixelRatio = pixelRatio;
    prevYaw = orbit.yaw;
    prevPitch = orbit.pitch;
    prevDistance = orbit.distance;
    prevTargetX = orbit.target[0];
    prevTargetY = orbit.target[1];
    prevTargetZ = orbit.target[2];
  };
}
