# Voxel Cone Tracing (VCT) — Integration Design for renderer3d_2

> Technical design document. Goal: add **hybrid VCT** (voxel cone tracing) global illumination (GI)
> to the existing WebGPU/WGSL renderer `renderer3d_2`. The document covers decisions, formulas, and
> their mapping to specific engine passes. No copy-paste-ready code is included — only prose,
> formulas, and short pseudo-snippets. Every significant choice is accompanied by a reference to
> the source repository/paper.

---

## 1. Overview and Final Recommendation

**What we are building.** A hybrid: primary visibility (what the camera sees per pixel) comes from
the existing G-buffer written by SDF impostors (sharp edges), while indirect light (light reflected
from other surfaces) is gathered by cone-tracing a voxel grid. This is exactly the multi-pass
scheme described by Wicked Engine ([voxelConeTracingHF.hlsli](https://github.com/turanszkij/WickedEngine/blob/master/WickedEngine/shaders/voxelConeTracingHF.hlsli))
and jose-villegas / VCTRenderer (deferred pass [light_pass.frag](https://github.com/jose-villegas/VCTRenderer/blob/master/engine/assets/shaders/light_pass.frag)).

**Isotropic first.** Start with **isotropic** voxels (one non-directional radiance volume + one
sample per cone) — the simplest working VCT (see
[Friduric/voxel-cone-tracing](https://github.com/Friduric/voxel-cone-tracing),
[Cigg/Voxel-Cone-Tracing](https://github.com/Cigg/Voxel-Cone-Tracing),
[Ramanjs/vxgi](https://github.com/Ramanjs/vxgi)). Anisotropic storage (6 directional volumes) is
kept as an upgrade against light leaking — implemented by
[rdinse/VCTGI](https://github.com/rdinse/VCTGI),
[jose-villegas](https://github.com/jose-villegas/VCTRenderer),
[HarshLight](https://github.com/MangoSister/HarshLight),
[DXE](https://github.com/LanLou123/DXE), and Wicked Engine.

**End-to-end pass list** in our engine (Z-up, reverse-Z, regular grid 128×128×32, cellSize 0.5):

1. **G-buffer** — already exists (SDF pass writes depth/normal/albedo).
2. **Voxelize + inject direct light** — a basic compute voxelize already exists (fills
   `voxelAlbedo.rgb=albedo, .a=occupancy`, `voxelEmission.rgb`). **Adding**: write of
   *direct-lit radiance* into a new volume `voxelRadiance` (see §4).
3. **3D texture mip pyramid** — **new** custom compute pass (isotropic opacity-weighted downsample;
   later — 6 directional convolutions). WebGPU has no `generateMipmap` for 3D (see §5).
4. **Cone-trace GI** — **new** fullscreen render pass over the G-buffer: 5–6 diffuse cones +
   optional 1 specular cone → writes indirect radiance (+ AO in alpha) into a half-resolution HDR
   texture (analogous to `gi`/`rc` in `createVoxelSystem`).
5. **Composite/resolve** — **new** (or extend the existing resolve): bilinear/bilateral upsample of
   indirect, addition with direct light, multiplication by AO and albedo (see §10).
6. *(optional)* **Multi-bounce / temporal** — re-injection of the previous frame's radiance into
   the voxel volume (§9).

---

## 1b. DECIDED ARCHITECTURE (current — supersedes exploratory text below)

> This section reflects the architecture actually decided after implementation + browser testing.
> Where it conflicts with later exploratory sections, **this wins.**

**Hard lesson (empirically confirmed in-browser): VCT is an INDIRECT-light technique only. Direct
light + its shadows MUST be computed separately and analytically.** We first tried a "pure unified"
model where ALL light (the sun and every emitter) was injected into the voxel volume and the diffuse
cone gather produced everything, including shadows. Result: a bright emitter's shadow does not darken
— it **fills/inverts** (the blocked cone gathers the occluder's own lit radiance + bounce, an
*average* of radiance, not a visibility test). This is intrinsic to the prefiltered diffuse gather
and is **not** fixed by anisotropy (verified: the inversion is identical with iso and aniso mips;
aniso only improves indirect colour/softness). Every reference renderer avoids this by structuring
lighting as **direct (analytic, shadowed) + VCT indirect** — confirmed in
[jose-villegas light_pass.frag](https://github.com/jose-villegas/VCTRenderer/blob/master/engine/assets/shaders/light_pass.frag)
(`compositeLighting = (directLighting + indirectLighting.rgb)·indirectLighting.a + emissive`),
[Crassin 2011](https://research.nvidia.com/sites/default/files/publications/GIVoxels-pg2011-authors.pdf),
[jose-villegas "Deferred Voxel Shading"](https://jose-villegas.github.io/post/deferred_voxel_shading/),
and §10 below.

**The decided model:**

1. **VCT cone gather = INDIRECT (bounce) ONLY.** Hemisphere FILL cones (Fibonacci, cosine-weighted)
   → indirect radiance (rgb) + AO (alpha). **No aimed-at-emitter cones** — the aimed cones were the
   "direct light via gather" that produced the filled/inverted shadows; they are removed.
2. **Each light is an ANALYTIC direct term:** `direct += lightColor·intensity·max(N·L,0)·falloff·visibility`.
   - Emitters (the `LightEmitter` spheres, incl. the "sun" emitter) are these analytic lights.
   - `falloff` is the inverse-square coefficient k (`1/(1+k·d²)`; k=0 = no falloff = a sun-like even light).
3. **Visibility = cone-traced soft shadow (Crassin).** One extra occlusion cone from the surface
   toward the light through the voxel volume; accumulate opacity; `visibility = 1 − opacity` (with
   distance falloff so distant occluders don't over-darken). Soft, cheap, reuses the voxels, **no
   shadow map needed.** (Crassin thesis "single cone to capture opacity";
   [Geeks3D](https://www.geeks3d.com/20121214/voxel-cone-tracing-global-illumination-in-opengl-4-3/),
   [Friduric](https://github.com/Friduric/voxel-cone-tracing).) A directional sun may instead reuse
   its existing **shadow map** (sunDepth pass) for a crisp cast shadow — both are standard.
4. **Composite:** `out = albedo·(ambient·AO + Σ direct + indirect) + emission` (matches jose-villegas /
   §10).
5. **Double-counting is avoided structurally:** the direct term is analytic; the gather is bounce
   only (FILL cones, started with a normal offset so it never samples the receiver's own voxel);
   removing the aimed cones removes the direct-emitter-via-gather path. Emitter *emission* still lives
   in the volume so it contributes to others' indirect bounce + its own glow (G-buffer emission), but
   not as a second copy of the direct term.
6. **Anisotropy (6 directional volumes, §3/§5) is KEPT** — it improves the indirect bounce
   (less leak, softer/cleaner colour). It does not (and cannot) fix direct-shadow correctness;
   that is what the analytic direct + visibility cone is for.

**Rationale.** We already have a regular grid and a compute voxelizer — this is exactly the target
architecture of Wicked Engine (which deliberately chose a regular grid over an SVO for fast
voxelization and cache-friendly marching — [Turánszki blog](https://wickedengine.net/2017/08/voxel-based-global-illumination/)).
The cone-trace is pure `textureSampleLevel` math and ports almost verbatim. The heaviest part of
classical VCT — geometry-shader conservative-rasterization voxelization — is not needed: we
already have compute (see §4).

---

## 2. What Already Exists in renderer3d_2 and What We Are Adding (Gap Analysis)

**Already exists** (files under `src/ECS/Systems/Lighting/`):

- `voxelResources.ts` — two 3D storage textures: `voxelAlbedo` (`rgba8unorm`,
  rgb=albedo, a=occupancy) and `voxelEmission` (`rgba16float`, rgb=emission). Grid: origin
  `(-32,-32,-2)`, cellSize 0.5, dims 128×128×32. Both `STORAGE_BINDING | TEXTURE_BINDING`.
- `voxelize.shader.ts` — compute `@workgroup_size(4,4,4)`, 1 thread = 1 voxel; voxel center
  → `scene_sdf` → if solid, `textureStore` albedo/occupancy/emission.
- `voxelTrace.wgsl.ts` — DDA (Amanatides–Woo) to the first solid voxel, `unproject` via
  `uInvViewProj`, `oct_decode`, `build_basis`, `trace_radiance`, `trace_interval`.
- G-buffer: `depthTexture` (reverse-Z), `normalTexture` (world-space normals),
  `albedoTexture`. World position is reconstructed via `invViewProj` (`unproject`).
- `createVoxelSystem.ts` — owns textures/pipelines/bind groups; existing passes:
  `voxelize` / `debug` / `gi` (brute-force) / `rc` (Radiance Cascades). Each pass has
  distinct uniform buffers; bind groups are rebuilt on resize/cellSize change.
- Conventions: `ShaderMeta` + `GPUShader` + `GPUVariable` + `wgsl` tag; `VariableKind.StorageTexture`
  already added (emitted as `texture_storage_3d<fmt, access>`).

**Adding for VCT:**

| What | Resource/pass type | Format |
|---|---|---|
| `voxelRadiance` — direct-lit radiance per voxel (mip 0) | 3D texture, `STORAGE` (inject write) + `TEXTURE` (mip/trace read) | `rgba16float` (HDR) |
| `voxelRadiance` mip pyramid (isotropic) | custom compute pass, one dispatch per level | `rgba16float`, separate view per mip |
| *(upgrade)* 6 directional radiance volumes with mips | 6× 3D textures (or one packed ×6 along X) | `rgba16float` |
| Inject direct light into `voxelRadiance` | extension of compute-voxelize **or** separate compute | — |
| Cone-trace GI pass | fullscreen render over G-buffer | output `rgba16float` |
| Composite/resolve | fullscreen render | output `bgra8unorm`/`rgba16float` |

**Key difference from the existing RC branches (`gi`/`rc`).** RC traces thin rays through a
*binary* grid (`occupancy`>0.5 → emission at first hit). VCT instead **samples the mip pyramid**
with a wide cone: a single `textureSampleLevel` at the appropriate LOD replaces dozens of march
steps. That is the core win: a pre-integrated volume instead of a full ray march.

---

## 3. Voxel Storage: Isotropic vs Anisotropic

### What to Store Per Voxel

A VCT cone needs **outgoing radiance** (light emitted/reflected by the voxel) + **opacity/coverage**
(how opaque the voxel is). Not albedo. Hence a separate `voxelRadiance` volume:

- `rgb` = direct-lit radiance (albedo · N·L · shadow · lightColor + emission).
- `a` = opacity/coverage (1 = fully solid voxel). Alpha serves both front-to-back
  accumulation and AO.

`voxelAlbedo`/`voxelEmission`/normals are needed only during the inject stage (see §4). This is the
design used by [jose-villegas](https://jose-villegas.github.io/post/deferred_voxel_shading/)
(three r32ui volumes albedo/normal/emission + a separate `voxelRadiance`) and
[DXE](https://github.com/LanLou123/DXE/blob/master/DXE/Shaders/radiance.hlsl) (5 isotropic
volumes: albedo/normal/emissive/radiance/flag).

### Isotropic vs Anisotropic — Comparison Table

| | Isotropic (1 volume) | Anisotropic (6 directions) |
|---|---|---|
| Storage | 1× `rgba16float` 3D + mips | 6× `rgba16float` 3D + mips |
| Cone sample | 1 `textureSampleLevel` | 3 samples (visible faces), blended by `abs(dir)` |
| Light leaking through walls | strong (mip averages occupancy end-to-end) | weak (directional occlusion) |
| Mip pass complexity | 1 convolution `0.125·Σ8` (or opacity-weighted) | 6 directional front-to-back convolutions |
| Memory for 128×128×32 | see below | ×6 |
| Repositories | Friduric, Cigg, Ramanjs, AlerianEmperor, maritim, sfreed141 | rdinse/VCTGI, jose-villegas, HarshLight, DXE, Wicked |

**Memory (our grid 128×128×32 = 524 288 voxels).** `rgba16float` = 8 bytes/voxel.
Base level = 4 MB. Full mip pyramid adds ≈ 1/7 ≈ +14% → ≈ **4.6 MB** for an isotropic
`voxelRadiance`. Anisotropic = **×6 ≈ 27 MB** (if all 6 volumes are at full resolution). Many
repos store the directional volumes at **half resolution** — jose-villegas: *"six 3D textures at
half resolution of the radiance volume"* — giving ×6·(1/8) ≈ ×0.75 of full resolution, i.e.
≈ 3.4 MB. For a cube of 128³ this would be more significant (Friduric 64³, Cigg 512³,
rdinse/DXE/HarshLight 256³), but with our Z=32 the volume is small — memory is not a concern
in either variant.

### Recommendation + Format

- **Start isotropic**: one `voxelRadiance` `rgba16float` with a mip chain. **✓ Decided (user):
  format `rgba16float`.** `rgba8unorm` (as used by Friduric/Cigg/Ramanjs) would also work, but
  we already have HDR emission (`rgba16float`), and temporal/multi-bounce accumulates values >1
  → `rgba16float` to avoid the unorm clamp.
- **Filtering**: linear+mip-linear for cone tracing (requires `texture_3d<f32>` + sampler with
  trilinear). `CLAMP_TO_EDGE` on all axes (our DDA grid is small; clamp-to-border/black as used
  by Friduric is an anti-leak measure, but our scene does not extend past the grid — edge is
  sufficient).
- **Upgrade to anisotropy** — when light leaking through thin walls becomes visible (§11). Implement
  as 6 separate `rgba16float` 3D textures (cleaner than the ×6-along-X packing used by
  [DXE/MeshVoxelizer.cpp](https://github.com/LanLou123/DXE/blob/master/DXE/Features/MeshVoxelizer.cpp));
  WebGPU has no limit on storage texture bindings that would block this.

---

## 4. Voxelization and Direct Light Injection

### Conservative Rasterization vs Compute — and Why We Use Compute

Classical VCT voxelizes via **rasterization with a geometry shader**: a triangle is projected along
its dominant axis into an N×N viewport to cover the maximum number of voxels; edges are optionally
expanded using the GPU Gems 2 ch.42 conservative rasterization method. This is the approach used by
[Friduric voxelization.geom](https://github.com/Friduric/voxel-cone-tracing/blob/master/Shaders/Voxelization/voxelization.geom),
[rdinse Voxelization.geom](https://github.com/rdinse/VCTGI/blob/master/src/shaders/VCTGI/Voxelization.geom),
[HarshLight voxelize_geom.glsl](https://github.com/MangoSister/HarshLight/blob/master/HarshLight/src/shaders/voxelize_geom.glsl),
[jose-villegas voxelization.frag](https://github.com/jose-villegas/VCTRenderer/blob/master/engine/assets/shaders/voxelization.frag),
and Wicked ([objectPS_voxelizer.hlsl](https://github.com/turanszkij/WickedEngine/blob/master/WickedEngine/shaders/objectPS_voxelizer.hlsl)).

**WebGPU does not support this:** no geometry shaders, no default hardware conservative
rasterization, no atomic operations on float textures. All sources that assess a WebGPU port
recommend **compute voxelization** — which is exactly what we already have. Our voxelizer does
not rasterize triangles; it **samples the SDF at each voxel center** (1 thread = 1 voxel,
`solid = d <= cellSize·0.5·√3`). This eliminates the entire conservative-raster problem: the SDF
provides analytic coverage, so gaps in thin surfaces cannot occur by construction. Atomics and
moving-average blending (required in raster paths because multiple fragments write to one voxel —
[rdinse imageAtomicRGBA8Avg](https://github.com/rdinse/VCTGI/blob/master/src/shaders/VCTGI/Voxelization.frag),
Wicked `InterlockedAdd`) are also unnecessary: each thread owns exactly one voxel, no races.

**Conclusion:** keep the compute voxelizer as-is. This is the cleanest path, and all sources
explicitly recommend it for WebGPU regular-grid setups.

### How to Inject Direct Light into a Voxel

Two variants, both appear in the literature:

**(A) Inject inside the voxelize pass** (Friduric, Cigg, AlerianEmperor, Ramanjs): for each solid
voxel, compute direct lighting immediately and write `voxelRadiance.rgb = albedo·(N·L)·shadow·lightColor
+ emission`, `a = occupancy`. Injection formula (Ramanjs
[voxelization.frag](https://github.com/Ramanjs/vxgi/blob/main/shaders/voxelization.frag)):
`lighting = (1-shadow)·max(dot(L,N),0)·lightColor·albedo + ke·emissive`.

**(B) Separate compute inject using a shadow map** (rdinse, jose-villegas
[inject_radiance.comp](https://github.com/jose-villegas/VCTRenderer/blob/master/engine/assets/shaders/inject_radiance.comp),
HarshLight [dirlight_injection_comp.glsl](https://github.com/MangoSister/HarshLight/blob/master/HarshLight/src/shaders/dirlight_injection_comp.glsl),
DXE [radiance.hlsl](https://github.com/LanLou123/DXE/blob/master/DXE/Shaders/radiance.hlsl)):
dispatch over shadow-map texels, unproject each into world space, reproject into the voxel grid,
`radiance = albedo · max(N·L,0) · lightColor`. Cheaper (dispatch over shadow-map, not over the
volume), but requires a shadow map.

**Recommendation. ✓ Decided (user): variant (A) + voxel-DDA shadow ray for sun visibility;
shadow-map injection (variant B) deferred as a possible future addition (see below).** We have a
single light source — `SunLight.ts` (directional). Shadow maps do not currently exist (the GI
branches use voxel-DDA for visibility). Therefore:

- **Stage 1 (chosen)**: variant (A) inside voxelize. We already have `emission_of(k)` and a face
  normal from the DDA; for the sun we compute `N·L` from the voxel normal (direction from the SDF
  gradient) and write `voxelRadiance.rgb = albedo·max(N·L,0)·sunColor·visibility + emission`,
  `a=occupancy`.
- **Sun visibility (chosen)**: **voxel-DDA shadow ray** from the voxel toward the sun (same as
  `trace_radiance`/`dda` in `voxelTrace.wgsl.ts`): `visibility = 1` if the ray exits the grid
  without hitting a solid voxel, otherwise `0` (or a softer fractional value based on how far the
  ray traveled). This reuses the existing DDA with no additional resources. Cost: +1 DDA per
  solid voxel in voxelize (once per frame, not per pixel — cheap).

Normal for `N·L`: take from the SDF gradient at the voxel center (`normalize(∇ scene_sdf)`) —
we already have the analytic SDF, which is more accurate than the averaged normals produced by
rasterization.

#### Possible Future Addition: Shadow-Map Injection (Variant B)

A future improvement, **not for the first version**. Once (and if) a **shadow map** exists for the
sun (a depth texture of the scene rendered from the light's camera), switch to variant (B): a
separate compute pass dispatches over shadow-map texels, unprojects each world position, reprojects
into the voxel grid, and writes `radiance = albedo·max(N·L,0)·sunColor`. This is the approach used by
[jose-villegas inject_radiance.comp](https://github.com/jose-villegas/VCTRenderer/blob/master/engine/assets/shaders/inject_radiance.comp),
[HarshLight dirlight_injection_comp.glsl](https://github.com/MangoSister/HarshLight/blob/master/HarshLight/src/shaders/dirlight_injection_comp.glsl),
[DXE radiance.hlsl](https://github.com/LanLou123/DXE/blob/master/DXE/Shaders/radiance.hlsl).
Advantages over the voxel-DDA shadow ray: (1) **cheaper** — dispatch over shadow-map texels rather
than over the entire volume with a ray per voxel; (2) **sharper/more accurate shadows** — no
voxel-stepping staircase artifacts; (3) scales to multiple light sources. Disadvantage: requires
the shadow-map pass itself (rendering the scene from the sun's camera) and world→voxel
reprojection. **Only the inject stage (§4) needs to change** — the cone trace, mips, and composite
remain as-is.

---

## 5. Building the 3D Texture Mip Pyramid in WebGPU

**Why a custom compute pass.** WebGPU has **no** built-in `generateMipmap` for 3D textures, and a
hardware box filter (used by Friduric/Cigg/Ramanjs/AlerianEmperor via `glGenerateMipmap(GL_TEXTURE_3D)`)
would average radiance and opacity end-to-end, ignoring direction → leaking. All serious ports build
mips in a **custom compute pass**:
[rdinse PreIntegration.comp](https://github.com/rdinse/VCTGI/blob/master/src/shaders/VCTGI/PreIntegration.comp),
[maritim voxelMipmapCompute.glsl](https://github.com/maritim/Voxel-Cone-Tracing/blob/master/Assets/Shaders/Voxelize/voxelMipmapCompute.glsl),
[sfreed141 filterRadiance.comp](https://github.com/sfreed141/vct/blob/master/shaders/filterRadiance.comp),
[HarshLight anisotropic_mipmap_*](https://github.com/MangoSister/HarshLight/blob/master/HarshLight/src/shaders/anisotropic_mipmap_start_comp.glsl),
[jose-villegas aniso_mipmapbase.comp](https://github.com/jose-villegas/VCTRenderer/blob/master/engine/assets/shaders/aniso_mipmapbase.comp).

### Isotropic Downsample (Stage 1) — Opacity-Weighted

Not a plain box average. maritim averages color only over **non-empty** children and **sums** alpha:

```
// for each of the 8 children: contribution = (a==0) ? 0 : 1
// finalColor = Σ rgb·contribution / Σ contribution ; alpha = Σ a
```
([maritim voxelMipmapCompute.glsl](https://github.com/maritim/Voxel-Cone-Tracing/blob/master/Assets/Shaders/Voxelize/voxelMipmapCompute.glsl)).
sfreed141 and Cigg/Ramanjs use a flat `value·0.125` (8 children) — simpler, but more leaking.
**Recommendation:** opacity-weighted per maritim — nearly free, noticeably less leaking and less
darkening from empty voxels.

### Anisotropic Downsample (Upgrade) — 6 Directional Front-to-Back Convolutions

For each of the 6 directions, the 2×2×2 block is accumulated **front-to-back along that direction's
axis** (the near voxel occludes the far voxel by its alpha), then divided by 4. For the +X
direction (rdinse):

```
dst = (v0 + v4·(1-v0.a) + v1 + v5·(1-v1.a) + v2 + v6·(1-v2.a) + v3 + v7·(1-v3.a)) / 4
```
([rdinse PreIntegration.comp](https://github.com/rdinse/VCTGI/blob/master/src/shaders/VCTGI/PreIntegration.comp),
identically in [jose-villegas aniso_mipmapbase.comp](https://github.com/jose-villegas/VCTRenderer/blob/master/engine/assets/shaders/aniso_mipmapbase.comp),
[HarshLight](https://github.com/MangoSister/HarshLight/blob/master/HarshLight/src/shaders/anisotropic_mipmap_start_comp.glsl)).
For -X the definition of "front" flips. This is Crassin's pre-integration — the primary anti-leak
mechanism.

### Implementation in Our Engine

- **Pass**: new compute shader, `@workgroup_size(4,4,4)` — all sources measure 4³ as optimal:
  rdinse reports 3.77 ms@4³ vs 21 ms@2³, 167 ms@1³; same result in maritim/sfreed141.
- **Per-level dispatch**: loop `level = 0..L-1`; at level `c`, bind mip `c` as a sampled view
  (read) and mip `c+1` as `texture_storage_3d<rgba16float, write>` (write). A WebGPU storage
  texture binds **one** mip level at a time (like `glBindImageTexture(level)`), so we need a
  **separate view per mip** and a **separate bind group per level**. A natural barrier exists
  between levels: each `dispatchWorkgroups` in its own (or sequential) compute pass is already
  ordered by the command encoder.
- **writeBuffer hazard between passes in a single encoder.** The dst-mip size is passed as a
  uniform. If each level uses a `writeBuffer` into the same buffer within one encoder, the value
  will be overwritten before execution. **Fix:** either a **distinct uniform buffer per level**
  (as we already do for RC cascades: `cascadeBuf[c]`), or compute `dstMipRes = baseDim >> (level+1)`
  directly in the shader from `uGridDims` and a `level` constant. We use a distinct buffer per
  level — consistent with the current `createVoxelSystem` style.
- **Cannot sample and write the same texture in one pass** — so we read mip `c` as sampled and
  write mip `c+1` as storage (different levels → different views, no hazard).

### Number of Levels

For our 128×128×32 base: `L = floor(log2(max(128,128,32))) + 1 = log2(128)+1 = 8` levels.
Along the short axis (Z=32), after 5 downsamples we reach 1; Z stays at 1 beyond that — downsample
each axis independently (as real engines clamp) or cap `maxLod = log2(32) = 5` for cone tracing.
In practice cones always clamp LOD anyway (Friduric `MIPMAP_HARDCAP 5.4`, DXE `VOXELMIPCOUNT 9`
with clamp), so **allocate 8 levels, clamp LOD to ~5–6 in the trace** to avoid the "glitchiness
of very high mips" (Friduric's phrasing).

---

## 6. Diffuse Cone Tracing: Cone Count, Apertures, Directions, Weights

Strong consensus across sources: **5–6 cones** over the hemisphere, aperture ≈ 60° full angle.

**Crassin's canonical set: 6 cones, aperture 60°** (`tan(half-angle)=tan30°≈0.577`): 1 along the
normal + 5 tilted 60° from the normal, uniformly spaced in azimuth. Cosine weights, sum ≈ 1
([Crassin 2011](https://research.nvidia.com/publication/2011-09_interactive-indirect-illumination-using-voxel-cone-tracing)).

Specific values from repositories:

- **6 cones, weights {0.25, 0.15×5}** (sum 1), ring at `y=0.5` (60° from normal),
  `tanHalfAngle=0.577` — Cigg ([standard.frag](https://github.com/Cigg/Voxel-Cone-Tracing/blob/master/shaders/standard.frag)),
  AlerianEmperor ([VoxelConeTracing.fs](https://github.com/AlerianEmperor/Voxel-Cone-Tracing/blob/main/Voxel_Cone_Tracing_Final/Shader/VoxelConeTracing.fs)),
  sfreed141 ([phong.frag](https://github.com/sfreed141/vct/blob/master/shaders/phong.frag), directions from simonstechblog).
- **6 cones, cosine weights {π/4, 3π/20×5}**, aperture `tan30°` — HarshLight
  ([ds_indirect_diffuse_frag.glsl](https://github.com/MangoSister/HarshLight/blob/master/HarshLight/src/shaders/ds_indirect_diffuse_frag.glsl)),
  DXE ([common.hlsl](https://github.com/LanLou123/DXE/blob/master/DXE/Shaders/common.hlsl)).
- **6 cones "hemi-dodecahedron", center weight 1.0 + 5 side weights 0.607**, apertures `2·0.5`
  and `2·0.549092` (diameter at distance 1) — rdinse
  ([ScreenFillGlobalIllumination.frag](https://github.com/rdinse/VCTGI/blob/master/src/shaders/VCTGI/ScreenFillGlobalIllumination.frag)).
- **5 cones**: 1 along normal + 4 diagonal, weights {1.0, 0.707×4}, `coneRatio=1` — maritim
  ([voxelConeTraceFragment.glsl](https://github.com/maritim/Voxel-Cone-Tracing/blob/master/Assets/Shaders/VoxelConeTrace/voxelConeTraceFragment.glsl));
  5 cones, weights {1,1,1}, aperture 0.767 — Ramanjs ([vct.frag](https://github.com/Ramanjs/vxgi/blob/main/shaders/vct.frag)).
- **9 cones** (1+4 side+4 corner, `CONE_SPREAD=0.325`, weight 1) — Friduric (the author notes
  "5 would be enough").
- **16 cones** near-uniform over the sphere, cosine weights, lower hemisphere discarded
  (`if cosTheta<=0 continue`) — Wicked (`DIFFUSE_CONE_COUNT=16`,
  `DIFFUSE_CONE_APERTURE=0.872665` rad) — a quality preset.

**Recommendation.** Use **6 cones** (1 along normal + 5 in a ring at `y≈0.5`/60°) with
**cosine weights {π/4, 3π/20×5}** (normalized by dividing by their sum), aperture `tan(half)=0.577`
(60° full angle). Build the cone basis using the existing `build_basis(normal)` from
`voxelTrace.wgsl.ts` (column 2 = normal), express cone directions in tangent space and rotate by
the matrix. 6 cones is the best quality/cost tradeoff (majority of sources; 16 in Wicked is a
high-quality preset — start with 6).

Pseudo-set (tangent space, normal = +Z of our basis as in `build_basis`):
```
cone[0] = normal               , w0 = π/4
cone[1..5] = basis · ringDir_i , wi = 3π/20    // 5 directions at 60° from normal
amount = Σ wi·traceCone(...) ; amount /= Σ wi
```

---

## 7. Cone Trace Loop

Canonical scheme (uniform across all sources; formulas from Crassin, confirmed by
jose-villegas/Friduric):

**Relationship to cellSize.** `voxelSize` = world-space voxel size = our `cellSize` (uniform
`uGridOrigin.w`). Everything (start offset, step, LOD) is expressed in `cellSize`.

**LOD selection by diameter.** At distance `dist`, cone diameter is
`diameter = max(voxelSize, 2·tan(aperture/2)·dist)`, and
```
LOD = log2(diameter / voxelSize)
```
(jose-villegas/Crassin; identical to rdinse `log2(diameter·voxelRes)`, sfreed141
`log2(max(1,2·coneRadius))`, DXE `log2(radius/VOXELSCALE)`, Cigg/AlerianEmperor
`log2(diameter/voxelWorldSize)`). Clamped to `maxLod` (see §5).

**Front-to-back accumulation** (premultiplied alpha, "over" operator):
```
c += (1 - alpha) · s.rgb ;   alpha += (1 - alpha) · s.a
```
(rdinse, jose-villegas, sfreed141, Cigg, AlerianEmperor, Wicked — all use this). Friduric/Ramanjs
use an artistic variant `acc += 0.075·ll·voxel·pow(1-voxel.a,2)` — convenient for isotropic
without alpha normalization, but the standard "over" is cleaner for HDR. **We use the standard
front-to-back.**

**Step size.** Grows with diameter to bound the number of steps:
```
step = max(diameter·0.5, voxelSize) ;   dist += step ;   diameter = 2·tan(aperture/2)·dist
```
(rdinse `max(diameter/2, voxelSize)`, Cigg `diameter·0.5`, sfreed141 `coneHeight += coneRadius`,
jose-villegas `dist += diameter·β`, β≤1). **We use `diameter·0.5`** (half-diameter — smoother, as
in Cigg).

**Start offset against self-intersection** (so the cone does not sample the surface's own voxel).
Two techniques, we use both:
- shift the **origin** along the normal: `startPos = P + normal · k·voxelSize`, `k≈1..2.5`
  (Wicked `P + N·voxelSize0`; rdinse `P + 2.5·N/voxelRes`; Cigg/AlerianEmperor `P + N·voxelSize`;
  sfreed141 `P + bias·N·scale`, bias 1.0);
- start `dist` not at zero but at `dist0 ≈ voxelSize` (Cigg/AlerianEmperor) or larger for wide
  cones (HarshLight `dist0 = voxelSize/half_tan`).

**Recommendation:** `startPos = P + normal·1.5·cellSize`, `dist0 = cellSize`. We already have
`normalBias` as a parameter (`giParams.normalBias`, `rcParams.normalBias`) — reuse it.

**Max distance.** Stop when `alpha >= 1` (saturation), when the ray exits the grid, or when
`dist > maxDist`. Our grid is small (extent ≈ 64×64×16); `maxDist` ≈ the grid diagonal
(Friduric `SQRT2` in a normalized cube; Wicked `vxgi.max_distance`). Use
`maxDist ≈ extent diagonal length` (≈ 90 units) as a parameter (`giParams.maxDist` already exists
at 24, sufficient for starting out). Early-out at `alpha>=0.95` (Cigg/AlerianEmperor) — a cheap
early termination.

**Optional empty-space skip via our SDF.** Wicked skips empty regions using the SDF:
`step = max(stepSize, sdf - diameter)` (large step when the nearest surface is far away). We have
`scene_sdf` — this is a free march accelerator; can be added during polish.

---

## 8. Specular Cone (Optional)

One **narrow** cone along the reflected direction `R = reflect(viewDir, normal)` (same
`traceCone`, same front-to-back logic). Aperture derived from roughness/glossiness:

- `aperture = roughness` directly (Wicked
  [voxelConeTracingHF.hlsli](https://github.com/turanszkij/WickedEngine/blob/master/WickedEngine/shaders/voxelConeTracingHF.hlsli));
- `aperture = roughness · π · 0.1` (sfreed141);
- `aperture = clamp(tan(π/2·(1-gloss)), 0.0174, π)` — smooth → tiny cone ≈1°, rough → wider
  (jose-villegas, DXE);
- fixed narrow `0.07` (~8°) — Cigg/AlerianEmperor (when no roughness map is available).

**Recommendation.** We do not currently have explicit roughness in `voxelEmission`/materials.
**Stage 1 specular**: fixed aperture `~0.07` (narrow mirror) as in Cigg. When roughness is
available in the material — `aperture = roughness` (Wicked, simplest). Start offset larger than
for diffuse (Friduric `8·voxelSize`; sfreed141 bias 1.7) to remove floor self-reflection.

---

## 9. Multi-Bounce and Temporal (Optional)

**Multi-bounce via re-injection.** Light is injected into voxels once (direct), cones gather one
bounce. A second bounce means writing the gathered indirect radiance back into the voxels and
rebuilding the mips. Two working recipes:

- **Separate propagation pass over voxels** (jose-villegas
  [inject_propagation.comp](https://github.com/jose-villegas/VCTRenderer/blob/master/engine/assets/shaders/inject_propagation.comp)):
  per-voxel cone trace of the volume against itself, result accumulated into `voxelRadiance` →
  the next per-pixel trace effectively gives a 2nd bounce.
- **Cross-frame re-injection** (Wicked): the GI result from the previous frame is blended into
  the light used during injection in the current frame — each frame adds +1 bounce, amortized.

**Temporal.** Wicked shifts the previous frame's voxel radiance according to camera motion (clipmap
reprojection,
[vxgi_offsetprevCS.hlsl](https://github.com/turanszkij/WickedEngine/blob/master/WickedEngine/shaders/vxgi_offsetprevCS.hlsl)).
sfreed141 uses a simple EMA in `transferVoxels.comp`: `mix(prev, cur, 1-decay)`.

**We already have temporal infrastructure** for brute-force `gi`: ping-pong `giAccum[0/1]`, EMA
`accumAlpha`, reset on camera movement (`moved`). VCT cone tracing is far less noisy than brute
force (it averages a mip rather than Monte Carlo rays), so:

- **Stage 1**: no temporal — VCT is already stable.
- **Optional denoise**: reuse the same ping-pong EMA as `gi` on the cone-trace output.
- **Optional 2nd bounce**: re-injection per Wicked (simplest: blend `giAccum[last]` into inject
  §4). This is net-new work, not included in the first version.

---

## 10. Compositing with Direct Light; AO from Alpha; GI Scale

**Composite** (deferred, over the G-buffer). Sources agree:
```
out = (directLighting + indirectDiffuse) · albedo + indirectSpecular · specColor
```
with an AO multiplier (jose-villegas: `compositeLighting = (direct + indirect.rgb)·indirect.a`;
sfreed141: `indirect.rgb *= ambientScale·albedo; color = indirect + direct + specular`;
DXE: `(diffuseCol·Alb·lit + directCol)·(1-spec) + spec·specCol·lit`).

**AO from cone alpha** — free. The accumulated `alpha` of the diffuse cones = coverage:
`occlusion = 1 - clamp(indirect.a, 0, 1)` (sfreed141, Wicked). With distance falloff so distant
occluders do not over-darken:
```
aoSample += ((1 - aoSample)·s.a) / (1 + dist·aoFalloff)
```
(rdinse, jose-villegas `f(r)=1/(1+λr)`, Cigg/AlerianEmperor `/(1+0.03·diameter)`). AO multiplies
the entire result (or is blended with a floor `min(aoFloor + ao, 1)` — rdinse
[ScreenFillCompositing.frag](https://github.com/rdinse/VCTGI/blob/master/src/shaders/VCTGI/ScreenFillCompositing.frag)).

**GI intensity scale** — a separate non-physical multiplier. All sources include a "magic" knob:
Friduric `DIFFUSE_INDIRECT_FACTOR 0.52`, Cigg `4.0`, Wicked `vxgi.intensity`, jose-villegas
`globalIlluminationAlpha`. **We already have `giStrength`** in `giParams`/`rcParams` — reuse it;
the `ambient` floor is also already present.

**Implementation in our engine.** Extend the existing resolve pattern (`resolveGroup0`): it already
reads the `albedoTex`/`normalTex`/`depthTex` G-buffer + an indirect texture (`probeIrr`/`probeBlur`).
For VCT: input is the cone-trace output (HDR `rgba16float`, half resolution), bilateral upsample
(by normal+depth, with `weightNormal`/`weightPlane` weights already present) to full resolution,
then composite per the formula above. Direct lighting (sun) is computed in the resolve from the
G-buffer normal.

---

## 11. Pitfalls and Fixes

| Problem | Symptom | Fix (source) |
|---|---|---|
| **Self-occlusion / self-intersection** | surface shadows/illuminates itself, acne artifacts | start the cone with a normal offset `P + N·k·voxelSize` (k≈1.5..2.5) + `dist0≈voxelSize`; specular — larger offset. Wicked, rdinse `2.5·N/voxelRes`, sfreed141 bias 1.0/1.7, Friduric `N·(1+4·ISQRT2)·VOXEL_SIZE`. **Our engine**: `normalBias` parameter. |
| **Light leaking through thin walls** | light bleeds through a wall ~1 voxel thick | (1) **anisotropic 6-directional mips** with opacity-weighted front-to-back — primary fix (Crassin, rdinse, jose-villegas, HarshLight, DXE, Wicked); (2) opacity-weighted isotropic downsample (maritim) — partial improvement; (3) LOD clamp (Friduric `MIPMAP_HARDCAP 5.4`); (4) larger start offset "removes bleeding from close surfaces" (Friduric). **Thin walls are possible in our grid (Z=32 is shallow) → anisotropy will likely be needed.** |
| **Voxel aliasing / flickering** | staircase artifacts, jitter on a coarse grid | trilinear + continuous LOD `textureSampleLevel` (all sources); opacity-weighted mips; our compute voxelizer is deterministic (no last-writer-wins flickering, unlike Friduric/Cigg/Ramanjs/AlerianEmperor where this occurs). |
| **Over-occlusion (double-counted alpha)** | GI too dark | cone weights sum to 1; proper front-to-back `(1-alpha)`, not additive; distance-falloff AO. |
| **Darkening from empty voxels in mip** | mip level darker than it should be | average color only over non-empty children (maritim `finalColor/contributionCount`). |
| **writeBuffer hazard between passes** | mip levels/cascades read stale uniform | distinct uniform buffer per pass/level (we already do this for `cascadeBuf[c]`). |
| **Sample + write same texture** | WebGPU validation error/UB | read mip `c` (sampled view), write mip `c+1` (storage view) — different views; ping-pong for temporal. |

---

## 12. Incremental Implementation Plan (Testable Intermediate Layers)

Following our incremental approach (like phases 2.1→2.4 in `voxel-gi-plan.md`): each layer is a
separate present-mode for A/B comparison.

1. **Layer 0 — `voxelRadiance` + inject direct light.** Add the `rgba16float` 3D volume; write
   `albedo·(N·L)·sun + emission` in voxelize. **Test:** debug mode like `voxelDebug`, but reading
   `voxelRadiance` — shows a shaded (N·L) voxel scene, not flat albedo.
2. **Layer 1 — debug mip visualization.** Custom compute mip pass (isotropic, opacity-weighted).
   Debug shader with a LOD slider: samples `voxelRadiance` at the selected level via
   `textureSampleLevel`. **Test:** sharp at LOD 0, smooth blur at high LODs; no black holes
   (opacity weighting is working).
3. **Layer 2 — 1 cone along the normal.** Fullscreen pass over the G-buffer (`P` via `unproject`,
   `N` from `normalTexture`): one cone along the normal, LOD by diameter, front-to-back.
   **Test:** soft ambient occlusion / bent-normal appearance; corners are darker.
4. **Layer 3 — full set of 6 cones + cosine weights.** **Test:** color bleeding appears
   (colored surfaces tint nearby geometry); compare against brute-force `gi` (reference).
5. **Layer 4 — composite.** Bilateral upsample of indirect + addition with direct light + AO from
   alpha + `giStrength`/`ambient`. **Test:** final image, A/B against `raw`/`gi`.
6. **Layer 5 (optional) — specular cone.** Narrow reflected cone. **Test:** blurry reflection of
   neighbors visible on smooth surfaces.
7. **Layer 6 (optional) — anisotropic mips.** 6 directional volumes + directional convolutions +
   3-face blend in the sample. **Test:** leaking through thin walls disappears (A/B on a scene
   with a 1-voxel-thick wall).
8. **Layer 7 (optional) — multi-bounce / temporal.** Re-injection / EMA. **Test:** second bounce
   brightens deep recesses; no flickering.

Each layer is kept as a present-mode (`voxel` / `vct1` / `vct` / …) alongside `raw`/`voxel`/
`gi`/`rc`, as already done in `demo.ts`.

---

## 13. Open Questions / Decisions

- **Decided:** `voxelRadiance` format = `rgba16float` (HDR + compatible with temporal; `rgba8unorm`
  like Friduric/Cigg would also work but clips values >1 and is incompatible with multi-bounce
  accumulation).
- **Decided:** direct light injection in the voxelize pass (variant A) with a voxel-DDA shadow ray
  for sun visibility; shadow-map injection (variant B) deferred as a possible future addition (§4).
- **Isotropic or anisotropic from the start?** Recommendation: isotropic, upgrade when leaking
  becomes visible. Z=32 (shallow grid) increases the risk of leaking through thin walls →
  anisotropy may be needed sooner.
- **Specular:** whether to enable it at all; fixed aperture vs roughness-driven (requires roughness
  in material).
- **maxLod / MIPMAP_HARDCAP** and aperture/weight values — tuned visually (these are "magic"
  numbers everywhere).
- **Multi-bounce/temporal:** necessary at all for a static showcase scene?
- **Does VCT replace the existing `gi`/`rc`** or coexist as an additional present-mode (for now
  — coexist, for A/B comparison)?

---

## 14. Sources

All read directly from repository source files (key files listed below). Where the paper itself
was inaccessible, that is noted.

- **Crassin et al. 2011, "Interactive Indirect Illumination Using Voxel Cone Tracing"** —
  [research.nvidia.com](https://research.nvidia.com/publication/2011-09_interactive-indirect-illumination-using-voxel-cone-tracing).
  The canonical reference: anisotropic 6-directional voxels, `LOD=log2(d/voxelSize)`, front-to-back,
  6 cones at 60°, 2 bounces. *The PDF itself was too large to fetch; formulas confirmed via
  Villegas/Friduric, who cite them verbatim.*
- **Wicked Engine — Turánszki, "Voxel-based GI"** —
  [article](https://wickedengine.net/2017/08/voxel-based-global-illumination/) *(original returned
  522/Cloudflare, archive.org blocked in the environment — formulas taken from live engine shaders)*; code:
  [voxelConeTracingHF.hlsli](https://github.com/turanszkij/WickedEngine/blob/master/WickedEngine/shaders/voxelConeTracingHF.hlsli),
  [ShaderInterop_VXGI.h](https://github.com/turanszkij/WickedEngine/blob/master/WickedEngine/shaders/ShaderInterop_VXGI.h),
  [objectPS_voxelizer.hlsl](https://github.com/turanszkij/WickedEngine/blob/master/WickedEngine/shaders/objectPS_voxelizer.hlsl),
  [vxgi_offsetprevCS.hlsl](https://github.com/turanszkij/WickedEngine/blob/master/WickedEngine/shaders/vxgi_offsetprevCS.hlsl).
  Closest to our architecture (regular grid, fully dynamic).
- **jose-villegas / VCTRenderer — "Deferred Voxel Shading for Real-Time GI"** —
  [article](https://jose-villegas.github.io/post/deferred_voxel_shading/); code:
  [voxelization.frag](https://github.com/jose-villegas/VCTRenderer/blob/master/engine/assets/shaders/voxelization.frag),
  [inject_radiance.comp](https://github.com/jose-villegas/VCTRenderer/blob/master/engine/assets/shaders/inject_radiance.comp),
  [inject_propagation.comp](https://github.com/jose-villegas/VCTRenderer/blob/master/engine/assets/shaders/inject_propagation.comp),
  [aniso_mipmapbase.comp](https://github.com/jose-villegas/VCTRenderer/blob/master/engine/assets/shaders/aniso_mipmapbase.comp),
  [aniso_mipmapvolume.comp](https://github.com/jose-villegas/VCTRenderer/blob/master/engine/assets/shaders/aniso_mipmapvolume.comp),
  [light_pass.frag](https://github.com/jose-villegas/VCTRenderer/blob/master/engine/assets/shaders/light_pass.frag).
  Anisotropic, 2 bounces, AO formula.
- **rdinse / VCTGI** (bachelor's thesis; bc3.moe project page *returned 522, repository read directly*) —
  [ScreenFillGlobalIllumination.frag](https://github.com/rdinse/VCTGI/blob/master/src/shaders/VCTGI/ScreenFillGlobalIllumination.frag),
  [PreIntegration.comp](https://github.com/rdinse/VCTGI/blob/master/src/shaders/VCTGI/PreIntegration.comp),
  [Voxelization.frag](https://github.com/rdinse/VCTGI/blob/master/src/shaders/VCTGI/Voxelization.frag),
  [ScreenFillCompositing.frag](https://github.com/rdinse/VCTGI/blob/master/src/shaders/VCTGI/ScreenFillCompositing.frag).
  Anisotropic, half-res GI + bilateral upsample, workgroup-size timings.
- **MangoSister / HarshLight** —
  [voxelize_frag.glsl](https://github.com/MangoSister/HarshLight/blob/master/HarshLight/src/shaders/voxelize_frag.glsl),
  [dirlight_injection_comp.glsl](https://github.com/MangoSister/HarshLight/blob/master/HarshLight/src/shaders/dirlight_injection_comp.glsl),
  [anisotropic_mipmap_start_comp.glsl](https://github.com/MangoSister/HarshLight/blob/master/HarshLight/src/shaders/anisotropic_mipmap_start_comp.glsl),
  [vct_grid_frag.glsl](https://github.com/MangoSister/HarshLight/blob/master/HarshLight/src/shaders/vct_grid_frag.glsl).
  Hybrid: isotropic leaf + 6-directional mips; opacity step-correction.
- **LanLou123 / DXE** (DX12) —
  [pix.hlsl](https://github.com/LanLou123/DXE/blob/master/DXE/Shaders/pix.hlsl),
  [common.hlsl](https://github.com/LanLou123/DXE/blob/master/DXE/Shaders/common.hlsl),
  [baseMipRadiance.hlsl](https://github.com/LanLou123/DXE/blob/master/DXE/Shaders/baseMipRadiance.hlsl),
  [radiance.hlsl](https://github.com/LanLou123/DXE/blob/master/DXE/Shaders/radiance.hlsl).
  Anisotropic (×6 along X), static/dynamic split, shadow-map inject.
- **sfreed141 / vct** ("warped VCT") —
  [phong.frag](https://github.com/sfreed141/vct/blob/master/shaders/phong.frag),
  [filterRadiance.comp](https://github.com/sfreed141/vct/blob/master/shaders/filterRadiance.comp),
  [injectRadiance.comp](https://github.com/sfreed141/vct/blob/master/shaders/injectRadiance.comp),
  [transferVoxels.comp](https://github.com/sfreed141/vct/blob/master/shaders/transferVoxels.comp).
  Isotropic, roughness→aperture, temporal EMA, all-compute except voxelization.
- **maritim / Voxel-Cone-Tracing** —
  [voxelConeTraceFragment.glsl](https://github.com/maritim/Voxel-Cone-Tracing/blob/master/Assets/Shaders/VoxelConeTrace/voxelConeTraceFragment.glsl),
  [voxelMipmapCompute.glsl](https://github.com/maritim/Voxel-Cone-Tracing/blob/master/Assets/Shaders/Voxelize/voxelMipmapCompute.glsl),
  [voxelBorderCompute.glsl](https://github.com/maritim/Voxel-Cone-Tracing/blob/master/Assets/Shaders/Voxelize/voxelBorderCompute.glsl).
  Isotropic opacity-weighted mip + border-fill anti-leak; workgroup timings.
- **Friduric / voxel-cone-tracing** (pedagogical Crassin reference) —
  [voxel_cone_tracing.frag](https://github.com/Friduric/voxel-cone-tracing/blob/master/Shaders/Voxel%20Cone%20Tracing/voxel_cone_tracing.frag),
  [voxelization.frag](https://github.com/Friduric/voxel-cone-tracing/blob/master/Shaders/Voxelization/voxelization.frag).
  Simplest isotropic, hardware mips, 9 cones, specular/refraction/shadow cones.
- **Cigg / Voxel-Cone-Tracing** —
  [standard.frag](https://github.com/Cigg/Voxel-Cone-Tracing/blob/master/shaders/standard.frag),
  [voxelization.frag](https://github.com/Cigg/Voxel-Cone-Tracing/blob/master/shaders/voxelization.frag).
  Clean 6-cone isotropic reference for the trace loop.
- **Ramanjs / vxgi** —
  [vct.frag](https://github.com/Ramanjs/vxgi/blob/main/shaders/vct.frag),
  [voxelization.frag](https://github.com/Ramanjs/vxgi/blob/main/shaders/voxelization.frag).
  Simplest VCT baseline, 5 cones.
- **AlerianEmperor / Voxel-Cone-Tracing** —
  [VoxelConeTracing.fs](https://github.com/AlerianEmperor/Voxel-Cone-Tracing/blob/main/Voxel_Cone_Tracing_Final/Shader/VoxelConeTracing.fs),
  [Voxelization.fs](https://github.com/AlerianEmperor/Voxel-Cone-Tracing/blob/main/Voxel_Cone_Tracing_Final/Shader/Voxelization.fs).
  128³ isotropic, 6+1 cones, adaptive `dist+=diameter`.
- **phonowiz / voxel-cone-tracing** (GL 4.1 / macOS, OpenCL mips) —
  [voxelConeTracing.frag](https://github.com/phonowiz/voxel-cone-tracing/blob/master/Shaders/Voxel%20Cone%20Tracing/voxelConeTracing.frag),
  [downsize.cl](https://github.com/phonowiz/voxel-cone-tracing/blob/master/Compute%20Shaders/downsize.cl).
  Useful as an example of "mips as an array of separate textures" (GL workaround, not needed for us).
