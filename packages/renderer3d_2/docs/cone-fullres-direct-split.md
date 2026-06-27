# Future work: full-res emitter direct + low-res shadow/bounce (decouple frequencies)

Status: **NOT implemented** — design note only. Captured 2026-06-27.

## Problem

The cone GI pass runs at a downscaled resolution (`coneScale` = 2/4/8, set in the GUI as
"cone resolution"). Lowering it is the biggest perf lever for heavy scenes, but at quarter/eighth
res the **edges (silhouettes) of small shapes get over-lit ("засвет граней")** when a bright
emitter passes near them.

### Why it happens (it is undersampling, NOT a voxel-anisotropy issue)

- At 1/8 res, one cone texel covers an **8×8 = 64 full-res pixel block**, and its value is computed
  for a **single** point of that block (mapped via `texCoord` in `voxelCone.shader.ts`).
- If a shape is smaller than the block, that single point lands on the **brightly lit face** near
  the emitter, so the bright value covers the whole block — including the shape's dark edges and the
  background behind it.
- The normal-aware bilateral upsample (`upsample_cone` in `voxelComposite.shader.ts`) cannot rescue
  it: all 4 nearest cone texels can be "bright face" texels (they are 8 px apart, the shape is
  smaller), so there is **no dark-edge tap to blend toward**. At half-res (taps 2 px apart) such a
  tap exists, so the artifact is mild there.

Root cause: the emitter **direct** term is **high-frequency** (sharp bright contact + crisp
shadow), while bounce/AO are **low-frequency**. Downscaling kills exactly the high-frequency part.

### Why anisotropic voxels do NOT help

Anisotropic voxels (6 directional components per voxel instead of one isotropic value) reduce light
leaking through thin walls and improve directional accuracy of the cone march. They address a
different axis entirely and have **no effect** on the low-res upsample edge bleed. Wrong tool.

## Proposed fix: split the frequencies

Decouple the emitter lighting into two parts computed at different resolutions:

1. **Direct magnitude → FULL res, in the composite.** The analytic emitter direct
   `full = ndl · Lj · atten` is cheap (no voxel march). Computing it per full-res pixel makes the
   bright contact use a full-res `ndl`, which **correctly falls to 0 at the silhouette** → edges no
   longer blow out, at any cone resolution.
2. **Shadow / occlusion + bounce → LOW res, in the cone pass (as today).** The shadow march (the
   expensive part) and the SH bounce stay downscaled. Shadows are low-frequency, so a low-res
   shadow reads as a slightly softer penumbra — far less visible than a bright over-lit rim.

Result: low-res (1/4, 1/8) becomes usable **without** edge over-lighting; perf is preserved because
the expensive shadow march stays cheap.

## Implementation sketch / cost

This is a **moderate refactor**, not a localized tweak:

- The composite needs the **emitter list** (`lights[]` + `lightColor[]`) passed to it (today only
  the cone pass has them) so it can recompute the full-res analytic direct.
- The cone pass must output a **shadow / visibility factor** rather than the final summed
  `directEmitters` rgb. The tricky part is the current **per-emitter** bleed-cancel logic
  (`contrib = full - max(0, shadow - bleed)`): aggregating multiple colored lights into a single
  upsamplable visibility channel is not linear, so this needs rethinking (e.g. output per-emitter
  shadow, or a weighted aggregate, or restrict the crisp path to the dominant light).
- Keep the existing aggregate path as a fallback for the bounce term.

Trade-off accepted: low-res shadows get softer. Acceptable because shadow is low-frequency.

## Related knobs already in place

- `coneScale` (GUI: Cone GI → cone resolution) — half / quarter / eighth.
- Normal-aware bilateral upsample (`upsample_cone`) — keeps edges crisp at half-res; insufficient at
  1/4–1/8 for sub-block-sized shapes (the reason for this note).
- Per-pixel light culling in the aimed loop — skips negligible emitters before the march.
