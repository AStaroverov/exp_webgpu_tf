/*
Radiance Cascades 3D
    Combining radiance cascades with radiosity/irradiance methods
    This implementation is not perfect, see comments below :)


Cascades
    Cubemap layout
        Traces 1 ray per pixel, smallest probe is a 2x2 quad
        Like radiosity and irradiance methods, probes/rays exist on the surface of geometry -> good memory scaling
        5 cascades are used in this shader
    Ray tracing
        Rays are traced with different lengths between cascades
        Their distribution causes the resolution in phi to be a semilinear function of theta -> bins_phi = 4 + 8*theta_bin_index
    Merging
        Cascades are merged by linearly interpolating along the parent rays
        Spatial interpolation is weighted based on local visibility, approximated by the probes traced rays
    Primary ray sampling
        Uses normal bilinear interpolation when reading the cascades


Flickering
    A consequence of (mainly) temporally merging cascades and static probe positions


Light leaking from interpolation
    This implementation samples the smallest probes using naive bilinear interpolation, leading
    to some light leaking close to geometry
        A solution is to use weighted interpolation here as well


This implementation is pretty simple and should be extended:
    Dynamic probe positions
        Moving/complex geometry/light can cause spikes/flickering with static probes
        A solution like RTXGI could work as an example
    Can be extended to integrate more complex BRDFs
        An idea is to importance sample both ray directions and probe positions based on screen space BRDF changes
        Otherwise hgih frequency detail would be blurred away


Feedback is welcome! :)


Controls
    Press and move mouse in x-axis to control camera
*/

vec4 TextureCube(vec2 uv, float lod) {
    //Samples the cubemap
    float tcSign = -mod(floor(uv.y*I1024), 2.)*2. + 1.;
    vec3 tcD = vec3(vec2(uv.x, mod(uv.y, 1024.))*I512 - 1., tcSign);
    if (uv.y > 4096.) tcD = tcD.xzy;
    else if (uv.y > 2048.) tcD = tcD.zxy;
    return textureLod(iChannel3, tcD, lod);
}

vec3 AcesFilm(vec3 x) {
    //Aces film curve
    return clamp((x*(2.51*x + 0.03))/(x*(2.43*x + 0.59) + 0.14), 0., 1.);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec3 Color = vec3(0.);
    vec3 sunDir = GetSunDirection(iTime);
    vec3 sunLight = GetSunLight(iTime);
    
    float nt = CYCLETIME_OFFSET + iTime*ICYCLETIME;
    if (iMouse.z > 0.) nt = nt*0. + (iMouse.x - iMouse.z)*2./(iResolution.x*ICYCLETIME);
    float cycleTime = (nt - CYCLETIME_OFFSET)*2.;
    vec3 pPos = vec3(0.5, 0.2, 0.2 + (cos(cycleTime)*0.5 + 0.5)*0.6);
    float sunA = nt*2.4;
    vec3 pEyeTarget = vec3(sin(sunA)*0.5 + 0.5, 0.1, cos(sunA)*0.5 + 0.5);
    vec3 pEye = normalize(pEyeTarget - pPos);
    vec3 pDir = normalize(vec3((fragCoord*IRES*2. - 1.)*ASPECT, 1.)*TBN(pEye));
    
    HIT rayHit = TraceRay(pPos, pDir, 1000000., iTime);
    if (rayHit.n.x > -15.) {
        if (rayHit.c.x < -1.5) {
            //Reflective
            vec3 rDir = reflect(pDir, rayHit.n);
            HIT rayHit2 = TraceRay(pPos + pDir*rayHit.t + rayHit.n*0.001, rDir, 1000000., iTime);
            if (rayHit2.n.x > -15.) {
                if (rayHit.c.x > 1.) {
                    //Emissive
                    Color += rayHit.c;
                } else if (dot(rayHit2.n, rDir) < 0.) {
                    vec2 suv = clamp(rayHit2.uv*128., vec2(0.5), rayHit2.res*0.5 - 0.5) + rayHit2.uvo;
                    Color = TextureCube(suv, 0.).xyz + TextureCube(suv + vec2(rayHit2.res.x*0.5, 0.), 0.).xyz +
                            TextureCube(suv + vec2(0., rayHit2.res.y*0.5), 0.).xyz + TextureCube(suv + rayHit2.res*0.5, 0.).xyz;
                    
                    //Sunlight
                    vec3 sPos = pPos + pDir*rayHit.t + rayHit.n*0.001 + rDir*rayHit2.t + rayHit2.n*0.001;
                    if (dot(rayHit2.n, sunDir) > 0.) {
                        if (TraceRay(sPos, sunDir, 10000., iTime).n.x < -1.5) Color += sunLight*dot(rayHit2.n, sunDir);
                    }

                    //Color
                    Color *= rayHit2.c;
                }
            } else {
                Color = GetSkyLight(rDir);
            }
        } else if (rayHit.c.x > 1.) {
            //Emissive
            Color += rayHit.c;
        } else {
            if (dot(rayHit.n, pDir) < 0.) {
                vec2 suv = clamp(rayHit.uv*128., vec2(0.5), rayHit.res*0.5 - 0.5) + rayHit.uvo;
                Color = TextureCube(suv, 0.).xyz + TextureCube(suv + vec2(rayHit.res.x*0.5, 0.), 0.).xyz +
                        TextureCube(suv + vec2(0., rayHit.res.y*0.5), 0.).xyz + TextureCube(suv + rayHit.res*0.5, 0.).xyz;
            
                //Sunlight
                vec3 sPos = pPos + pDir*rayHit.t + rayHit.n*0.001;
                if (dot(rayHit.n, sunDir) > 0.) {
                    if (TraceRay(sPos, sunDir, 10000., iTime).n.x < -1.5) Color += sunLight*dot(rayHit.n, sunDir);
                }

                //Color
                Color *= rayHit.c;
            }
        }
    } else {
        Color = GetSkyLight(pDir);
    }
    
    
    //Visualize cubemap
    //Color = TextureCube(floor(fragCoord) + 0.5 + vec2(0., max(0., iMouse.y - 100.)*IRES.y*2000.*1. + 243.*0.), 0.).xyz;
    
    
    fragColor = vec4(pow(AcesFilm(max(vec3(0.), Color)), vec3(0.45)), 1.);
}