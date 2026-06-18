import { ShaderMeta } from "renderer/src/WGSL/ShaderMeta.ts";
import { VariableKind, VariableMeta } from "renderer/src/Struct/VariableMeta.ts";
import { wgsl } from "renderer/src/WGSL/wgsl.ts";

export const RC_MAX_STEPS = 36;

export const shaderMeta = new ShaderMeta(
  {
    linearSampler: new VariableMeta("linearSampler", VariableKind.Sampler, `sampler`),
    sceneTexture: new VariableMeta("sceneTexture", VariableKind.Texture, `texture_2d<f32>`),
    distanceTexture: new VariableMeta("distanceTexture", VariableKind.Texture, `texture_2d<f32>`),
    lastTexture: new VariableMeta("lastTexture", VariableKind.Texture, `texture_2d<f32>`),
    emitDirTexture: new VariableMeta("emitDirTexture", VariableKind.Texture, `texture_2d<f32>`),
    resolution: new VariableMeta("uResolution", VariableKind.Uniform, `vec2<f32>`),
    cascadeCount: new VariableMeta("uCascadeCount", VariableKind.Uniform, `f32`),
    cascadeIndex: new VariableMeta("uCascadeIndex", VariableKind.Uniform, `f32`),
    baseRayCount: new VariableMeta("uBaseRayCount", VariableKind.Uniform, `f32`),
    rayInterval: new VariableMeta("uRayInterval", VariableKind.Uniform, `f32`),
    intervalOverlap: new VariableMeta("uIntervalOverlap", VariableKind.Uniform, `f32`),
    srgb: new VariableMeta("uSrgb", VariableKind.Uniform, `f32`),
    misc: new VariableMeta("uMisc", VariableKind.Uniform, `vec4<f32>`),
    sunAngle: new VariableMeta("uSunAngle", VariableKind.Uniform, `f32`),
    sunColor: new VariableMeta("uSunColor", VariableKind.Uniform, `vec4<f32>`),
    skyColor: new VariableMeta("uSkyColor", VariableKind.Uniform, `vec4<f32>`),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const POSITION = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0),
    vec2f(1.0, 1.0),
    vec2f(-1.0, 1.0)
  );

const TEX_COORDS = array<vec2f, 6>(
    vec2f(0.0, 1.0),
    vec2f(1.0, 1.0),
    vec2f(1.0, 0.0),
    vec2f(0.0, 1.0),
    vec2f(1.0, 0.0),
    vec2f(0.0, 0.0)
  );

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) texCoord: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var output: VertexOutput;
  output.position = vec4f(POSITION[vertexIndex], 0.0, 1.0);
  output.texCoord = TEX_COORDS[vertexIndex];
  return output;
}

const PI: f32 = 3.14159265;
const TAU: f32 = 2.0 * PI;
const basePixelsBetweenProbes: f32 = 1.0;
const cascadeInterval: f32 = 1.0;

fn fmod(a: f32, b: f32) -> f32 {
  return a - b * floor(a / b);
}

// Directional source (moon/sun/whatever: uSunColor.rgb, .w = sunDistance/softness)
// + faint sky fill (uSkyColor.rgb, .w = skyMix).
fn sunAndSky(rayAngle: f32) -> vec3f {
  let angleToSun = fmod(rayAngle - uSunAngle, TAU);
  let sunIntensity = pow(max(0.0, cos(angleToSun)), 4.0 / uSunColor.w);
  return mix(uSunColor.rgb * sunIntensity, uSkyColor.rgb, uSkyColor.w);
}

fn raymarch(rayStart: vec2f, rayEnd: vec2f, scale: f32, oneOverSize: vec2f, minStepSize: f32) -> vec4f {
  let rayDir = normalize(rayEnd - rayStart);
  let rayLength = length(rayEnd - rayStart);

  var rayUv = rayStart * oneOverSize;

  var dist = 0.0;
  for (var step = 0; step < ${RC_MAX_STEPS} && dist < rayLength; step = step + 1) {
    if (any(rayUv < vec2f(0.0)) || any(rayUv > vec2f(1.0))) {
      break;
    }

    let df = textureSampleLevel(distanceTexture, linearSampler, rayUv, 0.0).r;

    if (df <= minStepSize) {
      var sampleLight = textureSampleLevel(sceneTexture, linearSampler, rayUv, 0.0);
      sampleLight = vec4f(pow(sampleLight.rgb, vec3f(uSrgb)), sampleLight.a);

      // Directional emitter cone. textureLoad (NEAREST, no sampler) avoids blending
      // the unit facing with cleared (0,0) at emitter edges. length > 0.5 classifies
      // directional vs omni ((0,0)). Light travels emitter->receiver, i.e. -rayDir.
      let dirTexel = vec2<i32>(rayUv * uResolution);
      let emitDir  = textureLoad(emitDirTexture, dirTexel, 0).xy;
      if (length(emitDir) > 0.5) {
        let f = normalize(emitDir);
        sampleLight = vec4f(sampleLight.rgb * pow(max(0.0, dot(-rayDir, f)), uMisc.z), sampleLight.a);
      }

      // Alpha = hit opacity: per-material (emission.a = 1 - translucency) times
      // the global opacity (uMisc.w). < 1 lets merge() pull a share of farther
      // light (upper cascades) through the surface — translucent occluders.
      return vec4f(sampleLight.rgb, clamp(sampleLight.a, 0.0, 1.0) * uMisc.w);
    }

    dist += df * scale;
    rayUv += rayDir * (df * scale * oneOverSize);
  }

  return vec4f(0.0);
}

fn getUpperCascadeTextureUv(index: f32, offset: vec2f, spacingBase: f32) -> vec2f {
  let upperSpacing = pow(spacingBase, uCascadeIndex + 1.0);
  let upperSize = floor(uResolution / upperSpacing);
  let upperPosition = vec2f(
    fmod(index, upperSpacing),
    floor(index / upperSpacing)
  ) * upperSize;

  let clamped = clamp(offset, vec2f(0.5), upperSize - 0.5);
  return (upperPosition + clamped) / uResolution;
}

fn merge(currentRadiance: vec4f, index: f32, position: vec2f, spacingBase: f32, localOffset: vec2f) -> vec4f {
  // Alpha compositing: a miss (a = 0) passes the upper cascade through fully;
  // a hit with opacity a < 1 leaks (1 - a) of the farther light — translucency.
  let through = 1.0 - clamp(currentRadiance.a, 0.0, 1.0);
  if (through <= 0.0 || uCascadeIndex >= max(1.0, uCascadeCount - 1.0)) {
    return currentRadiance;
  }

  let offset = (position + localOffset) / spacingBase;

  let upperProbePosition = getUpperCascadeTextureUv(index, offset, spacingBase);

  let upperSample = textureSampleLevel(lastTexture, linearSampler, upperProbePosition, 0.0).rgb;

  return currentRadiance + vec4f(upperSample * through, through);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let vUv = input.texCoord;
  let coord = floor(vUv * uResolution);

  let base = uBaseRayCount;
  let rayCount = pow(base, uCascadeIndex + 1.0);
  let spacingBase = sqrt(uBaseRayCount);
  let spacing = pow(spacingBase, uCascadeIndex);

  let modifierHack = select(spacingBase, pow(basePixelsBetweenProbes, 1.0), base < 16.0);

  let size = floor(uResolution / spacing);
  let probeRelativePosition = vec2f(fmod(coord.x, size.x), fmod(coord.y, size.y));
  let rayPos = floor(coord / size);

  let modifiedInterval = modifierHack * uRayInterval * cascadeInterval;

  let start = select(pow(base, uCascadeIndex - 1.0), 0.0, uCascadeIndex == 0.0) * modifiedInterval;
  let end = ((1.0 + 3.0 * uIntervalOverlap) * pow(base, uCascadeIndex) - pow(uCascadeIndex, 2.0)) * modifiedInterval;

  let interval = vec2f(start, end);

  let probeCenter = (probeRelativePosition + 0.5) * basePixelsBetweenProbes * spacing;

  let preAvgAmt = uBaseRayCount;

  let baseIndex = (rayPos.x + (spacing * rayPos.y)) * preAvgAmt;
  let angleStep = TAU / rayCount;

  let scale = min(uResolution.x, uResolution.y);
  let oneOverSize = 1.0 / uResolution;
  let minStepSize = min(oneOverSize.x, oneOverSize.y) * 0.5;
  let avgRecip = 1.0 / preAvgAmt;

  var totalRadiance = vec4f(0.0);
  let noise = 0.0;

  for (var i = 0; i < i32(preAvgAmt); i = i + 1) {
    let index = baseIndex + f32(i);
    let angle = (index + 0.5 + noise) * angleStep;
    let rayDir = vec2f(cos(angle), -sin(angle));
    let rayStart = probeCenter + rayDir * interval.x;
    let rayEnd = rayStart + rayDir * interval.y;
    let raymarched = raymarch(rayStart, rayEnd, scale, oneOverSize, minStepSize);
    var mergedRadiance = merge(raymarched, index, probeRelativePosition, spacingBase, vec2f(0.5));

    if (uMisc.y > 0.5 && uCascadeIndex == uCascadeCount - 1.0) {
      mergedRadiance = vec4f(max(sunAndSky(angle), mergedRadiance.rgb), mergedRadiance.a);
    }

    totalRadiance += mergedRadiance * avgRecip;
  }

  let result = select(
    pow(totalRadiance.rgb, vec3f(1.0 / uSrgb)),
    totalRadiance.rgb,
    uCascadeIndex > uMisc.x
  );

  return vec4f(result, 1.0);
}
    `,
);
