import { mat4, vec3 } from "gl-matrix";

// Pure computation of the sun-POV orthographic view-projection for the shadow map: fit a tight ortho
// box to the camera-visible slice of the grid AABB, oriented along the sun direction. No GPU work —
// the caller uploads the result. Factory-with-scratch so there are ZERO per-frame allocations and the
// temporaries stay per-instance (no module-level sharing between voxel systems).

export type SunViewProjBox = {
  originX: number;
  originY: number;
  originZ: number;
  extentX: number;
  extentY: number;
  extentZ: number;
  cellSize: number;
};

export type SunViewProjResult = {
  // World units per shadow texel (larger ortho axis / map resolution) → composite normal-offset bias.
  sunWorldTexel: number;
  // Unit direction TOWARD the sun (same formula as voxelize()/composite()); the caller negates it
  // for the sun travel/ray direction.
  sdx: number;
  sdy: number;
  sdz: number;
};

export function createSunViewProjComputer() {
  const camInvViewProj = mat4.create();
  const sunView = mat4.create();
  const sunProj = mat4.create();
  const sunEye = vec3.create();
  const sunCenter = vec3.create();
  const sunUp = vec3.create();
  const sunCorner = vec3.create();
  const camCorner = vec3.create();
  const result: SunViewProjResult = { sunWorldTexel: 0, sdx: 0, sdy: 0, sdz: 0 };

  // Writes the sun view-projection (orthoZO) into `out` and returns the reused `result`.
  return function computeSunViewProj(
    box: SunViewProjBox,
    shadowSize: number,
    cameraViewProj: mat4,
    sun: { angle: number; elevation: number },
    out: mat4,
  ): SunViewProjResult {
    const { originX, originY, originZ, extentX, extentY, extentZ, cellSize } = box;
    // Grid AABB (world) — the clamp region + the depth range.
    const gMinX = originX;
    const gMinY = originY;
    const gMinZ = originZ;
    const gMaxX = originX + extentX;
    const gMaxY = originY + extentY;
    const gMaxZ = originZ + extentZ;

    // (A) Camera visible XY region: unproject the 8 reverse-Z NDC cube corners through the inverse
    // camera viewProj and clamp each into the grid box. The XY span of the clamped set is the
    // region the shadow map should cover at full resolution.
    mat4.invert(camInvViewProj, cameraViewProj);
    let wMinX = Infinity;
    let wMinY = Infinity;
    let wMaxX = -Infinity;
    let wMaxY = -Infinity;
    for (let i = 0; i < 8; i++) {
      camCorner[0] = i & 1 ? 1 : -1;
      camCorner[1] = i & 2 ? 1 : -1;
      camCorner[2] = i & 4 ? 1 : 0; // reverse-Z: near=1, far=0 → both clip planes
      vec3.transformMat4(camCorner, camCorner, camInvViewProj);
      const px = Math.min(Math.max(camCorner[0], gMinX), gMaxX);
      const py = Math.min(Math.max(camCorner[1], gMinY), gMaxY);
      if (px < wMinX) wMinX = px;
      if (px > wMaxX) wMaxX = px;
      if (py < wMinY) wMinY = py;
      if (py > wMaxY) wMaxY = py;
    }
    // Degenerate (camera doesn't overlap the grid / inverted matrix) → fall back to the full grid.
    if (!(wMaxX > wMinX) || !(wMaxY > wMinY)) {
      wMinX = gMinX;
      wMaxX = gMaxX;
      wMinY = gMinY;
      wMaxY = gMaxY;
    }
    // Pad XY so a caster just outside the view still casts its shadow tip into it (the XY fit is
    // clipped; the margin trades a little resolution for fewer popping edges), re-clamped to grid.
    const padX = (wMaxX - wMinX) * 0.15 + cellSize;
    const padY = (wMaxY - wMinY) * 0.15 + cellSize;
    wMinX = Math.max(gMinX, wMinX - padX);
    wMaxX = Math.min(gMaxX, wMaxX + padX);
    wMinY = Math.max(gMinY, wMinY - padY);
    wMaxY = Math.min(gMaxY, wMaxY + padY);

    // Region center: XY from the fitted view region, Z from the full grid (eye centered in depth).
    const cx = (wMinX + wMaxX) * 0.5;
    const cy = (wMinY + wMaxY) * 0.5;
    const cz = (gMinZ + gMaxZ) * 0.5;

    // Direction TOWARD the sun (unit), same formula as voxelize()/composite().
    const a = sun.angle;
    const e = sun.elevation;
    const ce = Math.cos(e);
    const sdx = Math.cos(a) * ce;
    const sdy = Math.sin(a) * ce;
    const sdz = Math.sin(e);

    // Eye pulled back from the region center along +dirTowardSun; the ortho near/far fit below
    // folds the distance out, so any value past the bounding-sphere radius is fine.
    const radius = 0.5 * Math.hypot(wMaxX - wMinX, wMaxY - wMinY, extentZ);
    const dist = radius * 2.0 + 1.0;
    sunCenter[0] = cx;
    sunCenter[1] = cy;
    sunCenter[2] = cz;
    sunEye[0] = cx + sdx * dist;
    sunEye[1] = cy + sdy * dist;
    sunEye[2] = cz + sdz * dist;

    // Up-vector degeneracy: when the sun is near-vertical (|sdz|→1) forward ∥ +Z makes lookAt
    // produce NaNs; swap up to +Y in that case.
    if (Math.abs(sdz) > 0.999) {
      sunUp[0] = 0;
      sunUp[1] = 1;
      sunUp[2] = 0;
    } else {
      sunUp[0] = 0;
      sunUp[1] = 0;
      sunUp[2] = 1;
    }
    mat4.lookAt(sunView, sunEye, sunCenter, sunUp);

    // Fit the ortho box IN LIGHT/VIEW space to the box [viewXY] × [full grid Z]: the full grid Z
    // is used for the corners' height so a tall caster's top (which under a tilted sun projects to
    // a different light-space XY than its base) is still inside the XY bounds. In view space the
    // camera looks down -Z, so visible z is negative.
    let lminX = Infinity;
    let lminY = Infinity;
    let lminZ = Infinity;
    let lmaxX = -Infinity;
    let lmaxY = -Infinity;
    let lmaxZ = -Infinity;
    for (let i = 0; i < 8; i++) {
      sunCorner[0] = i & 1 ? wMaxX : wMinX;
      sunCorner[1] = i & 2 ? wMaxY : wMinY;
      sunCorner[2] = i & 4 ? gMaxZ : gMinZ;
      vec3.transformMat4(sunCorner, sunCorner, sunView);
      if (sunCorner[0] < lminX) lminX = sunCorner[0];
      if (sunCorner[0] > lmaxX) lmaxX = sunCorner[0];
      if (sunCorner[1] < lminY) lminY = sunCorner[1];
      if (sunCorner[1] > lmaxY) lmaxY = sunCorner[1];
      if (sunCorner[2] < lminZ) lminZ = sunCorner[2];
      if (sunCorner[2] > lmaxZ) lmaxZ = sunCorner[2];
    }
    // near/far are POSITIVE distances; view-space z is negative going forward, so
    // near = -lmaxZ (closest), far = -lminZ (farthest). Pad to avoid front-face clipping.
    const near = -lmaxZ - 1.0;
    const far = -lminZ + 1.0;
    mat4.orthoZO(sunProj, lminX, lmaxX, lminY, lmaxY, near, far);
    mat4.multiply(out, sunProj, sunView);

    result.sunWorldTexel = Math.max(lmaxX - lminX, lmaxY - lminY) / shadowSize;
    result.sdx = sdx;
    result.sdy = sdy;
    result.sdz = sdz;
    return result;
  };
}
