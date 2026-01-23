import { ShaderMeta } from 'renderer/src/WGSL/ShaderMeta.ts';
import { VariableKind, VariableMeta } from 'renderer/src/Struct/VariableMeta.ts';
import { wgsl } from 'renderer/src/WGSL/wgsl.ts';

// Maximum number of VFX instances (combined for all effect types)
export const MAX_VFX_COUNT = 512;

// ============================================
// WGSL Code Fragments
// ============================================

const WGSL_CONSTANTS = /* wgsl */`
    const VFX_EXHAUST_SMOKE: u32 = 0u;
    const VFX_EXPLOSION: u32 = 1u;
    const VFX_HIT_FLASH: u32 = 2u;
    const VFX_MUZZLE_FLASH: u32 = 3u;
`;

const WGSL_VERTEX_OUTPUT = /* wgsl */`
    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) @interpolate(flat) instance_index: u32,
        @location(1) local_position: vec2f,
    };
`;

const WGSL_VERTEX_MAIN = /* wgsl */`
    @vertex
    fn vs_main(
        @builtin(vertex_index) vertex_index: u32,
        @builtin(instance_index) instance_index: u32,
    ) -> VertexOutput {
        let vfxInfo = uVFXData[instance_index];
        let maxRadius = vfxInfo.w;
        
        // Create quad vertices centered at origin
        let local_pos = vec2f(
            select(-maxRadius, maxRadius, vertex_index > 0u && vertex_index < 4u),
            select(-maxRadius, maxRadius, vertex_index > 1u && vertex_index < 5u)
        );
        
        // Transform to world/screen position
        let world_pos = (uProjection * uTransform[instance_index] * vec4f(local_pos, 0.0, 1.0)).xy;
        
        let position = vec4f(world_pos.x, -world_pos.y, 1.0, 1.0);

        return VertexOutput(
            position,
            instance_index,
            local_pos,
        );
    }
`;

const WGSL_UTILS = /* wgsl */`
    fn hash21(p: vec2f) -> f32 {
        var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
        p3 += dot(p3, p3.yzx + 19.19);
        return fract((p3.x + p3.y) * p3.z);
    }
    
    fn hash22(p: vec2f) -> vec2f {
        var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
        p3 += dot(p3, p3.yzx + 19.19);
        return fract(vec2f((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y));
    }
    
    fn noise2D(p: vec2f) -> f32 {
        let i = floor(p);
        let f = fract(p);
        let u = f * f * (3.0 - 2.0 * f);
        return mix(
            mix(hash21(i + vec2f(0.0, 0.0)), hash21(i + vec2f(1.0, 0.0)), u.x),
            mix(hash21(i + vec2f(0.0, 1.0)), hash21(i + vec2f(1.0, 1.0)), u.x),
            u.y
        );
    }
    
    fn fbm(p: vec2f, octaves: i32) -> f32 {
        var value = 0.0;
        var amplitude = 0.5;
        var frequency = 1.0;
        var pos = p;
        for (var i = 0; i < octaves; i++) {
            value += amplitude * noise2D(pos * frequency);
            amplitude *= 0.5;
            frequency *= 2.0;
        }
        return value;
    }
`;

// ============================================
// EXHAUST SMOKE EFFECT
// ============================================

const WGSL_EXHAUST_SMOKE = /* wgsl */`
    fn exhaustSmoke_pattern(localPos: vec2f, progress: f32, seed: f32) -> vec4f {
        let dist = length(localPos);
        let expandFactor = 1.0 + progress * 4.0;
        let currentSize = 1.0 * expandFactor;
        
        if (dist > currentSize) {
            return vec4f(0.0);
        }
        
        var turbulence = 0.0;
        let angle = atan2(localPos.y, localPos.x);
        
        for (var i = 0u; i < 5u; i++) {
            let freq = f32(i + 2u);
            let noiseAngle = angle * freq + seed * 10.0 + progress * f32(i) * 0.5;
            let noiseMag = hash21(vec2f(seed + f32(i) * 0.17, f32(i) * 0.31));
            turbulence += sin(noiseAngle) * noiseMag * (0.25 / freq);
        }
        
        let noisySize = currentSize * (1.0 + turbulence * 0.5);
        let normalizedDist = dist / noisySize;
        
        if (normalizedDist > 1.0) {
            return vec4f(0.0);
        }
        
        let colorVariation = hash21(vec2f(seed * 3.0, progress * 0.1));
        let baseGray = 0.25 + colorVariation * 0.1;
        let smokeColor = vec3f(baseGray, baseGray * 0.95, baseGray * 0.9);
        
        let edgeFalloff = 1.0 - smoothstep(0.2, 0.9, normalizedDist);
        let fadeOut = 1.0 - smoothstep(0.1, 0.7, progress);
        let centerDensity = 1.0 - normalizedDist * 0.4;
        let alpha = edgeFalloff * fadeOut * centerDensity * 0.5;
        
        return vec4f(smokeColor, alpha);
    }

    fn exhaustSmoke_wisps(localPos: vec2f, progress: f32, seed: f32) -> vec4f {
        let wispCount = 4u;
        var totalAlpha = 0.0;
        var wispColor = vec3f(0.3, 0.28, 0.26);
        
        for (var i = 0u; i < wispCount; i++) {
            let wispSeed = hash21(vec2f(seed * 20.0 + f32(i), f32(i) * 7.0));
            let wispAngle = f32(i) * 1.57 + wispSeed * 0.8;
            let driftSpeed = 0.8 + wispSeed * 0.5;
            let wispDist = progress * driftSpeed * (1.0 + progress * 1.5);
            
            let wispPos = vec2f(cos(wispAngle), sin(wispAngle)) * wispDist;
            let distToWisp = length(localPos - wispPos);
            let wispSize = (0.4 + progress * 0.6) * (1.0 - progress * 0.7);
            
            if (distToWisp < wispSize) {
                let fadeOut = 1.0 - smoothstep(0.2, 0.8, progress);
                let wispAlpha = (1.0 - distToWisp / wispSize) * fadeOut * 0.3;
                totalAlpha = max(totalAlpha, wispAlpha);
            }
        }
        
        return vec4f(wispColor, totalAlpha);
    }

    fn renderExhaustSmoke(localPos: vec2f, progress: f32, seed: f32) -> vec4f {
        var finalColor = vec3f(0.0);
        var totalAlpha = 0.0;
        
        let smoke = exhaustSmoke_pattern(localPos, progress, seed);
        if (smoke.a > 0.0) {
            finalColor = smoke.rgb;
            totalAlpha = smoke.a;
        }
        
        let wisps = exhaustSmoke_wisps(localPos, progress, seed);
        if (wisps.a > 0.0) {
            finalColor = mix(finalColor, wisps.rgb, wisps.a * 0.5);
            totalAlpha = max(totalAlpha, wisps.a);
        }
        
        return vec4f(finalColor, totalAlpha);
    }
`;

// ============================================
// EXPLOSION EFFECT
// ============================================

const WGSL_EXPLOSION = /* wgsl */`
    fn explosion_fireball(localPos: vec2f, size: f32, progress: f32, seed: f32) -> vec4f {
        let dist = length(localPos);
        let expandProgress = min(progress * 3.0, 1.0);
        let currentSize = size * (0.3 + expandProgress * 0.7);
        
        if (dist > currentSize) {
            return vec4f(0.0);
        }
        
        let angle = atan2(localPos.y, localPos.x);
        var edgeNoise = 0.0;
        for (var i = 0u; i < 6u; i++) {
            let noiseAngle = angle * (f32(i) + 3.0) + seed * 10.0 + progress * 2.0;
            let noiseMag = hash21(vec2f(seed + f32(i) * 0.1, f32(i)));
            edgeNoise += sin(noiseAngle) * noiseMag * 0.15;
        }
        
        let noisySize = currentSize * (1.0 + edgeNoise);
        let normalizedDist = dist / noisySize;
        
        if (normalizedDist > 1.0) {
            return vec4f(0.0);
        }
        
        let innerColor = vec3f(1.0, 0.95, 0.7);
        let midColor = vec3f(1.0, 0.6, 0.1);
        let outerColor = vec3f(0.8, 0.2, 0.0);
        
        var color: vec3f;
        if (normalizedDist < 0.3) {
            color = mix(innerColor, midColor, normalizedDist / 0.3);
        } else {
            color = mix(midColor, outerColor, (normalizedDist - 0.3) / 0.7);
        }
        
        let centerIntensity = 1.0 - normalizedDist * 0.5;
        let fadeOut = 1.0 - smoothstep(0.3, 1.0, progress);
        let intensity = centerIntensity * fadeOut;
        
        return vec4f(color, intensity);
    }

    fn explosion_smoke(localPos: vec2f, size: f32, progress: f32, seed: f32) -> vec4f {
        if (progress < 0.1) {
            return vec4f(0.0);
        }
        
        let smokeProgress = (progress - 0.1) / 0.9;
        let dist = length(localPos);
        let ringRadius = size * (0.5 + smokeProgress * 0.8);
        let ringWidth = size * 0.4 * (1.0 - smokeProgress * 0.3);
        let ringDist = abs(dist - ringRadius);
        
        if (ringDist > ringWidth) {
            return vec4f(0.0);
        }
        
        let angle = atan2(localPos.y, localPos.x);
        var noise = 0.0;
        for (var i = 0u; i < 4u; i++) {
            let noiseAngle = angle * (f32(i) + 2.0) + seed * 5.0;
            noise += sin(noiseAngle + smokeProgress * 3.0) * 0.2;
        }
        
        let smokeIntensity = (1.0 - ringDist / ringWidth) * (1.0 - smokeProgress);
        let finalIntensity = smokeIntensity * (0.8 + noise * 0.2);
        let smokeColor = vec3f(0.3, 0.28, 0.25) * (1.0 - smokeProgress * 0.5);
        
        return vec4f(smokeColor, max(0.0, finalIntensity * 0.6));
    }

    fn explosion_debris(localPos: vec2f, size: f32, progress: f32, seed: f32) -> vec4f {
        let dist = length(localPos);
        let angle = atan2(localPos.y, localPos.x);
        
        var debrisIntensity = 0.0;
        var debrisColor = vec3f(1.0, 0.7, 0.3);
        
        let debrisCount = 16u;
        for (var i = 0u; i < debrisCount; i++) {
            let debrisSeed = hash21(vec2f(seed * 50.0 + f32(i), f32(i) * 13.0));
            let debrisAngle = f32(i) * 6.28318 / f32(debrisCount) + debrisSeed * 0.8;
            let speed = size * (1.5 + debrisSeed * 1.5);
            let debrisDist = progress * speed;
            let debrisPos = vec2f(cos(debrisAngle), sin(debrisAngle)) * debrisDist;
            let distToDebris = length(localPos - debrisPos);
            let debrisSize = size * 0.08 * (1.0 - progress * 0.7);
            
            if (distToDebris < debrisSize) {
                let brightness = (1.0 - distToDebris / debrisSize) * (1.0 - progress);
                debrisIntensity = max(debrisIntensity, brightness);
                let colorMix = hash21(vec2f(debrisSeed, seed));
                debrisColor = mix(vec3f(1.0, 0.9, 0.4), vec3f(1.0, 0.5, 0.1), colorMix);
            }
        }
        
        return vec4f(debrisColor, debrisIntensity);
    }

    fn explosion_flash(localPos: vec2f, size: f32, progress: f32, seed: f32) -> vec4f {
        if (progress > 0.15) {
            return vec4f(0.0);
        }
        
        let flashProgress = progress / 0.15;
        let dist = length(localPos);
        let flashSize = size * (0.5 + flashProgress * 1.0);
        
        if (dist > flashSize) {
            return vec4f(0.0);
        }
        
        let intensity = (1.0 - dist / flashSize) * (1.0 - flashProgress * flashProgress);
        return vec4f(1.0, 1.0, 1.0, intensity);
    }

    fn renderExplosion(localPos: vec2f, progress: f32, seed: f32) -> vec4f {
        let size = 1.0;
        var finalColor = vec3f(0.0);
        var totalAlpha = 0.0;
        
        let smoke = explosion_smoke(localPos, size, progress, seed);
        if (smoke.a > 0.0) {
            finalColor = mix(finalColor, smoke.rgb, smoke.a);
            totalAlpha = max(totalAlpha, smoke.a);
        }
        
        let fireball = explosion_fireball(localPos, size, progress, seed);
        if (fireball.a > 0.0) {
            finalColor = mix(finalColor, fireball.rgb, fireball.a);
            totalAlpha = max(totalAlpha, fireball.a);
        }
        
        let debris = explosion_debris(localPos, size, progress, seed);
        if (debris.a > 0.0) {
            finalColor = mix(finalColor, debris.rgb, debris.a);
            totalAlpha = max(totalAlpha, debris.a);
        }
        
        let flash = explosion_flash(localPos, size, progress, seed);
        if (flash.a > 0.0) {
            finalColor = mix(finalColor, flash.rgb, flash.a);
            totalAlpha = max(totalAlpha, flash.a);
        }
        
        return vec4f(finalColor, totalAlpha);
    }
`;

// ============================================
// HIT FLASH EFFECT
// ============================================

const WGSL_HIT_FLASH = /* wgsl */`
    fn hitFlash_sparks(localPos: vec2f, size: f32, progress: f32, seed: f32) -> f32 {
        let scaledSize = size * 2.0;
        let dist = length(localPos);
        let angle = atan2(localPos.y, localPos.x);
        let expandSpeed = scaledSize * 2.0;
        let sparkTravel = progress * expandSpeed;
        var sparkIntensity = 0.0;
        
        let sparkCount = 12u;
        for (var i = 0u; i < sparkCount; i++) {
            let sparkSeed = hash21(vec2f(seed * 100.0 + f32(i), f32(i) * 17.0));
            let sparkAngle = f32(i) * 6.28318 / f32(sparkCount) + sparkSeed * 0.5;
            let sparkLength = scaledSize * (0.8 + hash21(vec2f(sparkSeed, seed)) * 1.2);
            let sparkStartDist = scaledSize * 0.2;
            let sparkEndDist = sparkStartDist + sparkLength * progress;
            let angleDiff = abs(atan2(sin(angle - sparkAngle), cos(angle - sparkAngle)));
            let sparkWidth = scaledSize * 0.12 * (1.0 - progress * 0.5);
            
            if (angleDiff < sparkWidth / max(dist, 0.1)) {
                if (dist > sparkStartDist && dist < sparkEndDist) {
                    let alongSpark = (dist - sparkStartDist) / (sparkEndDist - sparkStartDist);
                    let brightness = (1.0 - alongSpark * 0.7) * (1.0 - progress * 0.8);
                    sparkIntensity = max(sparkIntensity, brightness);
                }
            }
        }
        
        return sparkIntensity;
    }

    fn hitFlash_impact(localPos: vec2f, size: f32, progress: f32, seed: f32) -> f32 {
        let scaledSize = size * 2.0;
        
        if (progress > 0.4) {
            return 0.0;
        }
        
        let flashProgress = progress / 0.4;
        let dist = length(localPos);
        let ringRadius = scaledSize * 0.4 * (1.0 + flashProgress * 2.0);
        let ringWidth = scaledSize * 0.5 * (1.0 - flashProgress * 0.5);
        var flash = 0.0;
        
        let coreSize = scaledSize * 0.6 * (1.0 - flashProgress * 0.6);
        if (dist < coreSize) {
            flash = 1.0 - smoothstep(0.0, coreSize, dist) * 0.5;
        }
        
        let ringDist = abs(dist - ringRadius);
        if (ringDist < ringWidth) {
            let ringIntensity = (1.0 - ringDist / ringWidth) * 0.9;
            flash = max(flash, ringIntensity);
        }
        
        flash *= 1.0 - flashProgress * flashProgress * 0.7;
        return flash;
    }

    fn renderHitFlash(localPos: vec2f, progress: f32, seed: f32) -> vec4f {
        let size = 1.0;
        var totalAlpha = 0.0;
        var finalColor = vec3f(0.0);
        
        let sparkIntensity = hitFlash_sparks(localPos, size, progress, seed);
        let flashIntensity = hitFlash_impact(localPos, size, progress, seed);
        
        if (flashIntensity > 0.0) {
            let flashColor = vec3f(1.0, 0.95, 0.8);
            finalColor = mix(finalColor, flashColor, flashIntensity);
            totalAlpha = max(totalAlpha, flashIntensity);
        }
        
        if (sparkIntensity > 0.0) {
            let sparkColor = mix(vec3f(1.0, 0.9, 0.5), vec3f(1.0, 0.6, 0.2), progress);
            let alpha = min(sparkIntensity * 1.2, 1.0);
            finalColor = mix(finalColor, sparkColor, alpha);
            totalAlpha = max(totalAlpha, alpha);
        }
        
        return vec4f(finalColor, totalAlpha);
    }
`;

// ============================================
// MUZZLE FLASH EFFECT
// ============================================

const WGSL_MUZZLE_FLASH = /* wgsl */`
    fn muzzleFlash_smoke(localPos: vec2f, size: f32, progress: f32, seed: f32) -> f32 {
        let expandedSize = size * (0.5 + progress * 1.8);
        var stretchedPos = localPos;
        stretchedPos.x *= 1.8;
        stretchedPos.y *= 0.6;
        stretchedPos.y += expandedSize * 0.5;
        
        let turbulenceScale = 3.0 + progress * 2.0;
        let turbulence = fbm(stretchedPos * turbulenceScale + vec2f(seed * 10.0, progress * 5.0), 4);
        stretchedPos += (turbulence - 0.5) * 0.3 * progress;
        
        let dist = length(stretchedPos);
        var density = 0.0;
        
        if (dist < expandedSize) {
            density = 1.0 - smoothstep(0.0, expandedSize, dist);
            let edgeNoise = fbm(stretchedPos * 5.0 + vec2f(seed * 20.0), 3);
            density *= 0.7 + edgeNoise * 0.6;
        }
        
        let puffCount = 6u;
        for (var i = 0u; i < puffCount; i++) {
            let fi = f32(i);
            let puffSeed = hash21(vec2f(seed * 100.0 + fi, fi * 3.7));
            let angle = fi * 6.28318 / f32(puffCount) + puffSeed * 1.5 - 1.57;
            let puffDist = expandedSize * 0.35 * (0.4 + progress * 0.6) * (0.7 + puffSeed * 0.6);
            let puffOffset = vec2f(cos(angle) * 0.6, sin(angle)) * puffDist;
            let puffSize = expandedSize * (0.25 + puffSeed * 0.2);
            let puffDistFromCenter = length(stretchedPos - puffOffset);
            
            if (puffDistFromCenter < puffSize) {
                let puffDensity = (1.0 - smoothstep(0.0, puffSize, puffDistFromCenter)) * 0.4;
                density = max(density, puffDensity);
            }
        }
        
        let fadeProgress = min(progress * 1.2, 1.0);
        density *= 1.0 - fadeProgress * fadeProgress * fadeProgress;
        
        return clamp(density, 0.0, 1.0);
    }

    fn muzzleFlash_flash(localPos: vec2f, size: f32, progress: f32, seed: f32) -> vec4f {
        if (progress > 0.2) {
            return vec4f(0.0);
        }
        
        let flashProgress = progress / 0.2;
        let flashSize = size * 1.3 * (1.0 - flashProgress * 0.4);
        
        var offsetPos = localPos;
        offsetPos.y += size * 0.25;
        
        let dist = length(offsetPos);
        let angle = atan2(offsetPos.x, -offsetPos.y);
        
        var intensity = 0.0;
        var coreIntensity = 0.0;
        
        let spikeCount = 5.0 + floor(hash21(vec2f(seed, 0.5)) * 4.0);
        let starAngle = angle * spikeCount;
        
        var totalSpikePattern = 0.0;
        for (var layer = 0u; layer < 3u; layer++) {
            let layerF = f32(layer);
            let layerSeed = hash21(vec2f(seed * 50.0 + layerF, layerF * 2.3));
            let layerAngle = starAngle + layerSeed * 3.14159;
            let spikeIndex = floor((angle + 3.14159) / (6.28318 / spikeCount));
            let randomVariation = hash21(vec2f(seed * 100.0 + spikeIndex + layerF * 10.0, spikeIndex * 7.0 + layerF));
            let spikeMultiplier = 0.4 + randomVariation * 1.0;
            let spike = pow(max(cos(layerAngle), 0.0), 1.2 + layerF * 0.5);
            totalSpikePattern += spike * spikeMultiplier * (1.0 - layerF * 0.25);
        }
        totalSpikePattern /= 2.0;
        
        let baseRadius = flashSize * 0.2;
        let spikeRadius = flashSize * 0.7;
        let starRadius = baseRadius + totalSpikePattern * spikeRadius;
        let edgeNoise = hash21(vec2f(angle * 10.0 + seed * 30.0, dist * 5.0));
        let noisyRadius = starRadius * (0.85 + edgeNoise * 0.3);
        
        if (dist < noisyRadius) {
            intensity = 1.0 - smoothstep(0.0, noisyRadius * 0.8, dist);
        }
        
        let coreSize = flashSize * 0.3;
        if (dist < coreSize) {
            coreIntensity = 1.0 - smoothstep(0.0, coreSize, dist);
            coreIntensity = pow(coreIntensity, 0.7);
        }
        
        let glowSize = flashSize * 1.5;
        var glowIntensity = 0.0;
        if (dist < glowSize) {
            glowIntensity = (1.0 - smoothstep(0.0, glowSize, dist)) * 0.3;
        }
        
        intensity = max(intensity, glowIntensity);
        let fade = 1.0 - flashProgress * flashProgress;
        intensity *= fade;
        coreIntensity *= fade;
        
        var color = vec3f(1.0, 0.85, 0.4);
        color = mix(color, vec3f(1.0, 0.95, 0.8), intensity * 0.5);
        color = mix(color, vec3f(1.0, 1.0, 1.0), coreIntensity);
        let edgeFactor = 1.0 - intensity;
        color = mix(color, vec3f(1.0, 0.5, 0.2), edgeFactor * 0.3 * (1.0 - coreIntensity));
        
        return vec4f(color, max(intensity, coreIntensity));
    }

    fn muzzleFlash_sparks(localPos: vec2f, size: f32, progress: f32, seed: f32) -> vec4f {
        if (progress > 0.4) {
            return vec4f(0.0);
        }
        
        let sparkProgress = progress / 0.4;
        var offsetPos = localPos;
        offsetPos.y += size * 0.2;
        
        var totalIntensity = 0.0;
        var sparkColor = vec3f(1.0, 0.9, 0.6);
        
        let sparkCount = 12u;
        for (var i = 0u; i < sparkCount; i++) {
            let fi = f32(i);
            let sparkSeed = hash22(vec2f(seed * 200.0 + fi, fi * 5.3));
            let spreadAngle = (sparkSeed.x - 0.5) * 1.2 - 1.57;
            let sparkSpeed = 0.5 + sparkSeed.y * 1.5;
            let sparkDist = sparkSpeed * sparkProgress * size * 2.5;
            let sparkPos = vec2f(cos(spreadAngle) * sparkDist, sin(spreadAngle) * sparkDist);
            let distToSpark = length(offsetPos - sparkPos);
            let sparkSize = size * 0.08 * (1.0 - sparkProgress * 0.7);
            
            if (distToSpark < sparkSize) {
                let sparkIntensity = (1.0 - smoothstep(0.0, sparkSize, distToSpark));
                let sparkFade = 1.0 - pow(sparkProgress, 0.5 + sparkSeed.y);
                totalIntensity = max(totalIntensity, sparkIntensity * sparkFade);
            }
        }
        
        sparkColor = mix(sparkColor, vec3f(1.0, 0.7, 0.3), hash21(vec2f(seed, totalIntensity)) * 0.4);
        return vec4f(sparkColor, totalIntensity);
    }

    fn renderMuzzleFlash(localPos: vec2f, progress: f32, seed: f32) -> vec4f {
        let size = 1.0;
        var finalColor = vec3f(0.0);
        var totalAlpha = 0.0;
        
        let smokeDensity = muzzleFlash_smoke(localPos, size, progress, seed);
        if (smokeDensity > 0.0) {
            let baseGray = mix(0.8, 0.45, progress);
            let warmth = max(0.0, 1.0 - progress * 3.0) * 0.15;
            let smokeColor = vec3f(baseGray + warmth, baseGray * 0.95 + warmth * 0.5, baseGray * 0.9);
            let alpha = smokeDensity * 0.45;
            finalColor = mix(finalColor, smokeColor, alpha);
            totalAlpha = max(totalAlpha, alpha);
        }
        
        let sparks = muzzleFlash_sparks(localPos, size, progress, seed);
        if (sparks.a > 0.0) {
            finalColor = mix(finalColor, sparks.rgb, sparks.a);
            totalAlpha = max(totalAlpha, sparks.a);
        }
        
        let flash = muzzleFlash_flash(localPos, size, progress, seed);
        if (flash.a > 0.0) {
            finalColor = mix(finalColor, flash.rgb, flash.a);
            totalAlpha = max(totalAlpha, flash.a);
        }
        
        return vec4f(finalColor, totalAlpha);
    }
`;

// ============================================
// FRAGMENT MAIN
// ============================================

const WGSL_FRAGMENT_MAIN = /* wgsl */`
    @fragment
    fn fs_main(
        @location(0) @interpolate(flat) instance_index: u32,
        @location(1) local_position: vec2f,
    ) -> @location(0) vec4f {
        let vfxInfo = uVFXData[instance_index];
        let progress = vfxInfo.x;
        let seed = vfxInfo.y;
        let effectType = u32(vfxInfo.z);
        
        var result: vec4f;
        
        switch (effectType) {
            case VFX_EXHAUST_SMOKE: {
                result = renderExhaustSmoke(local_position, progress, seed);
            }
            case VFX_EXPLOSION: {
                result = renderExplosion(local_position, progress, seed);
            }
            case VFX_HIT_FLASH: {
                result = renderHitFlash(local_position, progress, seed);
            }
            case VFX_MUZZLE_FLASH: {
                result = renderMuzzleFlash(local_position, progress, seed);
            }
            default: {
                result = vec4f(1.0, 0.0, 1.0, 1.0); // Debug: magenta for unknown type
            }
        }
        
        if (result.a < 0.01) {
            discard;
        }
        
        return result;
    }
`;

// ============================================
// SHADER META
// ============================================

export const shaderMeta = new ShaderMeta(
    {
        projection: new VariableMeta('uProjection', VariableKind.Uniform, `mat4x4<f32>`),
        transform: new VariableMeta('uTransform', VariableKind.StorageRead, `array<mat4x4<f32>, ${MAX_VFX_COUNT}>`),
        vfxData: new VariableMeta('uVFXData', VariableKind.StorageRead, `array<vec4<f32>, ${MAX_VFX_COUNT}>`),
    },
    {},
    wgsl`
        ${WGSL_CONSTANTS}
        ${WGSL_VERTEX_OUTPUT}
        ${WGSL_VERTEX_MAIN}
        ${WGSL_UTILS}
        ${WGSL_EXHAUST_SMOKE}
        ${WGSL_EXPLOSION}
        ${WGSL_HIT_FLASH}
        ${WGSL_MUZZLE_FLASH}
        ${WGSL_FRAGMENT_MAIN}
    `,
);
