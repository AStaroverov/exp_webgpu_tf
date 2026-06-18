//Cascades and merging

vec4 TextureCube(vec2 uv) {
    //Samples the cubemap
    float tcSign = -mod(floor(uv.y*I1024), 2.)*2. + 1.;
    vec3 tcD = vec3(vec2(uv.x, mod(uv.y, 1024.))*I512 - 1., tcSign);
    if (uv.y > 4096.) tcD = tcD.xzy;
    else if (uv.y > 2048.) tcD = tcD.zxy;
    return textureLod(iChannel3, tcD, 0.);
}

vec4 TextureCube(vec2 uv, float lod) {
    //Samples the cubemap
    float tcSign = -mod(floor(uv.y*I1024), 2.)*2. + 1.;
    vec3 tcD = vec3(vec2(uv.x, mod(uv.y, 1024.))*I512 - 1., tcSign);
    if (uv.y > 4096.) tcD = tcD.xzy;
    else if (uv.y > 2048.) tcD = tcD.zxy;
    return textureLod(iChannel3, tcD, lod);
}

vec4 WeightedSample(vec2 luvo, vec2 luvd, vec2 luvp, vec2 uvo, vec3 probePos,
                    vec3 gTan, vec3 gBit, vec3 gPos, float lProbeSize) {
    //Approximate probe visibility weighting (flatland assumption)
    vec3 lastProbePos = gPos + gTan*(luvp.x*lProbeSize/256.) + gBit*(luvp.y*lProbeSize/256.);
    vec3 relVec = probePos - lastProbePos;
    float theta = (lProbeSize*0.5 - 0.5)/(lProbeSize*0.5)*3.141592653*0.5;
    float phi = atan(-dot(relVec, gTan), -dot(relVec, gBit));
    float phiI = floor((phi/3.141592653*0.5 + 0.5)*(4. + 8.*(lProbeSize*0.5 - 1.))) + 0.5;
    vec2 phiUV;
    float phiLen = lProbeSize - 1.;
    if (phiI < phiLen) phiUV = vec2(lProbeSize - 0.5, lProbeSize - phiI);
    else if (phiI < phiLen*2.) phiUV = vec2(lProbeSize - (phiI - phiLen), 0.5);
    else if (phiI < phiLen*3.) phiUV = vec2(0.5, phiI - phiLen*2.);
    else phiUV = vec2(phiI - phiLen*3., lProbeSize - 0.5);
    float lProbeRayDist = TextureCube(luvo + floor(phiUV)*uvo + luvp).w;
    if (lProbeRayDist < -0.5 || length(relVec) < lProbeRayDist*cos(3.141592653*0.5 - theta) + 0.01) {
        vec2 luv = luvo + luvd + clamp(luvp, vec2(0.5), uvo - 0.5);;
        return vec4(TextureCube(luv).xyz + TextureCube(luv + vec2(uvo.x, 0.)).xyz +
                    TextureCube(luv + vec2(0., uvo.y)).xyz + TextureCube(luv + uvo).xyz, 1.);
    }
    return vec4(0.);
}

void mainCubemap(out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir) {
    vec4 Output = texture(iChannel3, rayDir);
    vec2 UV; vec3 aDir = abs(rayDir);
    if (aDir.z > max(aDir.x, aDir.y)) {
        //Z-side
        UV = floor(((rayDir.xy/aDir.z)*0.5 + 0.5)*1024.) + 0.5;
        if (rayDir.z < 0.) UV.y += 1024.;
    } else if (aDir.x > aDir.y) {
        //X-side
        UV = floor(((rayDir.yz/aDir.x)*0.5 + 0.5)*1024.) + 0.5;
        if (rayDir.x > 0.) UV.y += 2048.;
        else UV.y += 3072.;
    } else {
        //Y-side
        UV = floor(((rayDir.xz/aDir.y)*0.5 + 0.5)*1024.) + 0.5;
        if (rayDir.y > 0.) UV.y += 4096.;
        else UV.y += 5120.;
    }
    if (DFBox(UV, vec2(1024., (256.*6.)*2.)) < 0.) {
        //Cascades
        Output = vec4(0.); 
        vec2 gRes;
        vec3 gTan, gBit, gNor, gPos;
        
        //Hardcoded geometry
        if (UV.y < 256.*6.) {
            if (UV.x < 256.) {
                gTan = vec3(1., 0., 0.);
                gBit = vec3(0., 0., 1.);
                gNor = vec3(0., 1., 0.);
                gPos = vec3(0., 0., 0.);
                gRes = vec2(256.);
            } else if (UV.x < 512.) {
                gTan = vec3(1., 0., 0.);
                gBit = vec3(0., 0., 1.);
                gNor = vec3(0., -1., 0.);
                gPos = vec3(0., 0.5, 0.);
                gRes = vec2(256.);
            } else if (UV.x < 640.) {
                gTan = vec3(0., 1., 0.);
                gBit = vec3(0., 0., 1.);
                gNor = vec3(1., 0., 0.);
                gPos = vec3(0., 0., 0.);
                gRes = vec2(128., 256.);
            } else if (UV.x < 768.) {
                gTan = vec3(0., 1., 0.);
                gBit = vec3(0., 0., 1.);
                gNor = vec3(-1., 0., 0.);
                gPos = vec3(1., 0., 0.);
                gRes = vec2(128., 256.);
            } else if (UV.x < 896.) {
                gTan = vec3(0., 1., 0.);
                gBit = vec3(1., 0., 0.);
                gNor = vec3(0., 0., 1.);
                gPos = vec3(0., 0., 0.);
                gRes = vec2(128., 256.);
            } else {
                gTan = vec3(0., 1., 0.);
                gBit = vec3(1., 0., 0.);
                gNor = vec3(0., 0., -1.);
                gPos = vec3(0., 0., 1.);
                gRes = vec2(128., 256.);
            }
        } else {
            if (UV.x < 128.) {
                gTan = vec3(0., 1., 0.);
                gBit = vec3(1., 0., 0.);
                gNor = vec3(0., 0., -1.);
                gPos = vec3(0., 0., 0.47 - 1./256.);
                gRes = vec2(128., 256.);
            } else if (UV.x < 256.) {
                gTan = vec3(0., 1., 0.);
                gBit = vec3(1., 0., 0.);
                gNor = vec3(0., 0., 1.);
                gPos = vec3(0., 0., 0.53 - 1./256.);
                gRes = vec2(128., 256.);
            }
        }
        
        //Probe ray distribution
        vec3 sunDir = GetSunDirection(iTime);
        vec3 sunLight = GetSunLight(iTime);
        vec2 modUV = mod(UV, gRes);
        float probeCascade = floor(mod(UV.y, 1536.)/256.);
        float probeSize = pow(2., probeCascade + 1.);
        vec2 probePositions = gRes/probeSize;
        vec3 probePos = gPos + mod(modUV.x, probePositions.x)*probeSize/256.*gTan +
                               mod(modUV.y, probePositions.y)*probeSize/256.*gBit;
        vec2 probeUV = floor(modUV/probePositions) + 0.5;
        vec2 probeRel = probeUV - probeSize*0.5;
        float probeThetai = max(abs(probeRel.x), abs(probeRel.y));
        float probeTheta = probeThetai/probeSize*3.14192653;
        float probePhi = 0.;
        if (probeRel.x + 0.5 > probeThetai && probeRel.y - 0.5 > -probeThetai) {
            probePhi = probeRel.x - probeRel.y;
        } else if (probeRel.y - 0.5 < -probeThetai && probeRel.x - 0.5 > -probeThetai) {
            probePhi = probeThetai*2. - probeRel.y - probeRel.x;
        } else if (probeRel.x - 0.5 < -probeThetai && probeRel.y + 0.5 < probeThetai) {
            probePhi = probeThetai*4. - probeRel.x + probeRel.y;
        } else if (probeRel.y + 0.5 > probeThetai && probeRel.x + 0.5 < probeThetai) {
            probePhi = probeThetai*8. - (probeRel.y - probeRel.x);
        }
        probePhi = probePhi*3.141592653*2./(4. + 8.*floor(probeThetai));
        vec3 probeDir = vec3(vec2(sin(probePhi), cos(probePhi))*sin(probeTheta), cos(probeTheta));
        probeDir = probeDir.x*gTan + probeDir.y*gBit + probeDir.z*gNor;
        
        //RT
        float tInterval = (1./64.)*probeSize*2.;
        if (probeCascade > 4.5) tInterval = 10000.;
        HIT rayHit = TraceRay(probePos + gNor*0.001, probeDir, tInterval, iTime);
        if (rayHit.n.x > -1.5) {
            Output.w = rayHit.t;
            if (rayHit.c.x < -1.5) {
                //Reflective
                    //Do nothing
            } else if (rayHit.c.x > 1.) {
                //Emissive
                Output.xyz += rayHit.c;
            } else {
                //Geo
                if (dot(rayHit.n, probeDir) < 0.) {
                    //*
                    //Bounce light
                    vec2 suv = clamp(rayHit.uv*128., vec2(0.5), rayHit.res*0.5 - 0.5) + rayHit.uvo;
                    Output.xyz = TextureCube(suv, 0.).xyz + TextureCube(suv + vec2(rayHit.res.x*0.5, 0.), 0.).xyz +
                                 TextureCube(suv + vec2(0., rayHit.res.y*0.5), 0.).xyz + TextureCube(suv + rayHit.res*0.5, 0.).xyz;
                    //*/
                    
                    //Sunlight
                    vec3 sNor = rayHit.n;
                    vec3 sPos = probePos + gNor*0.001 + probeDir*rayHit.t + sNor*0.001;
                    if (dot(sNor, sunDir) > 0.) {
                        if (TraceRay(sPos, sunDir, 10000., iTime).n.x < -1.5) Output.xyz += sunLight*dot(sNor, sunDir);
                    }

                    //Color
                    Output.xyz *= rayHit.c;
                }
            }
        } else {
            //Sky
            Output.w = -1.;
            Output.xyz = GetSkyLight(probeDir);
        }
        
        //Hemisphere normalized area and BRDF
        Output.xyz *= (cos(probeTheta - 3.141592653/probeSize) -
                       cos(probeTheta + 3.141592653/probeSize))/(4. + 8.*floor(probeThetai));
        Output.xyz *= cos(probeTheta); //Diffuse
        
        //*
        //Merging with weighted bilinear
        if (probeCascade < 4.5) {
            float interpMinDist = (1./256.)*probeSize*1.5;
            float interpMaxInterval = interpMinDist;
            if (probeCascade < 0.5) { interpMinDist = 0.; interpMaxInterval *= 2.; }
            float l = 1. - clamp((rayHit.t - interpMinDist)/interpMaxInterval, 0., 1.);
            vec2 uvo = probePositions*0.5;
            vec2 lPUVOrigin = floor(UV/gRes)*gRes + vec2(0., gRes.y);
            vec2 lPUVDirs = floor(modUV/probePositions)*probePositions;
            vec2 lPUVPos = clamp(mod(modUV, probePositions)*0.5, vec2(0.5), uvo - 0.5);
            vec2 fPUVPos = fract(lPUVPos - 0.5);
            vec2 flPUVPos = floor(lPUVPos - 0.5) + 0.5;
            vec4 S0 = WeightedSample(lPUVOrigin, lPUVDirs, flPUVPos,
                                     uvo, probePos, gTan, gBit, gPos, probeSize*2.);
            vec4 S1 = WeightedSample(lPUVOrigin, lPUVDirs, flPUVPos + vec2(1., 0.),
                                     uvo, probePos, gTan, gBit, gPos, probeSize*2.);
            vec4 S2 = WeightedSample(lPUVOrigin, lPUVDirs, flPUVPos + vec2(0., 1.),
                                     uvo, probePos, gTan, gBit, gPos, probeSize*2.);
            vec4 S3 = WeightedSample(lPUVOrigin, lPUVDirs, flPUVPos + 1.,
                                     uvo, probePos, gTan, gBit, gPos, probeSize*2.);
            vec3 lastOutput = mix(mix(S0.xyz, S1.xyz, fPUVPos.x), mix(S2.xyz, S3.xyz, fPUVPos.x), fPUVPos.y)/
                              max(0.01, mix(mix(S0.w, S1.w, fPUVPos.x), mix(S2.w, S3.w, fPUVPos.x), fPUVPos.y));
            if (!isnan(lastOutput.x)) //TMP fix
                Output.xyz = Output.xyz*l + lastOutput*(1. - l);
        }
        //*/
        
        /*
        //Merging with normal bilinear
        if (probeCascade < 4.5) {
            float interpMinDist = (1./256.)*probeSize*1.5;
            float interpMaxInterval = interpMinDist;
            if (probeCascade < 0.5) { interpMinDist = 0.; interpMaxInterval *= 2.; }
            float l = 1. - clamp((rayHit.t - interpMinDist)/interpMaxInterval, 0., 1.);
            vec2 uvo = probePositions*0.5;
            vec2 lastProbeUV = floor(UV/gRes)*gRes + vec2(0., gRes.y) + 
                               floor(modUV/probePositions)*probePositions +
                               clamp(mod(modUV, probePositions)*0.5, vec2(0.5), uvo - 0.5);
            vec4 lastOutput = TextureCube(lastProbeUV) + TextureCube(lastProbeUV + vec2(uvo.x, 0.)) +
                              TextureCube(lastProbeUV + vec2(0., uvo.y)) + TextureCube(lastProbeUV + uvo);
            Output.xyz = Output.xyz*l + lastOutput.xyz*(1. - l);
        }
        //*/
    }
    fragColor = Output;
}