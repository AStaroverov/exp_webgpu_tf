import { mat4, quat, vec3 } from "gl-matrix";

export type Pose = {
  tx: number;
  ty: number;
  tz: number;
  rx: number;
  ry: number;
  rz: number;
};

const scratchPos = vec3.create();
const scratchRot = quat.create();
const scratchScale = vec3.create();

// quat -> intrinsic XYZ Euler in DEGREES, the inverse of gl-matrix quat.fromEuler
// (which is XYZ-degrees). Aliasing/gimbal mean a posed quaternion can read back as a
// different-looking Euler triple than was typed; that is acceptable per the prototype's
// numeric-pose scope.
function quatToEulerDeg(q: quat, out: Pose): void {
  const [x, y, z, w] = q;
  const sinP = 2 * (w * y - z * x);
  const pitch = Math.abs(sinP) >= 1 ? (Math.sign(sinP) * Math.PI) / 2 : Math.asin(sinP);
  const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  const deg = 180 / Math.PI;
  out.rx = roll * deg;
  out.ry = pitch * deg;
  out.rz = yaw * deg;
}

export function readPose(m: mat4, out: Pose): void {
  mat4.getTranslation(scratchPos, m);
  out.tx = scratchPos[0];
  out.ty = scratchPos[1];
  out.tz = scratchPos[2];
  mat4.getRotation(scratchRot, m);
  quatToEulerDeg(scratchRot, out);
}

export function writePose(m: mat4, p: Pose): void {
  mat4.getScaling(scratchScale, m);
  quat.fromEuler(scratchRot, p.rx, p.ry, p.rz);
  scratchPos[0] = p.tx;
  scratchPos[1] = p.ty;
  scratchPos[2] = p.tz;
  mat4.fromRotationTranslationScale(m, scratchRot, scratchPos, scratchScale);
}
