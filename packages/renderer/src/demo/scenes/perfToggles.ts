import type GUI from "lil-gui";

// Per-pass toggles shared by BOTH perf scenes ("perf", "perf2") AND the demo render loop: the
// loop runs each pass purely by its toggle, so flipping one changes ONLY that pass's GPU work
// and the GPU-ms delta attributes its cost. A module singleton (like the DI objects) because
// the scene owns the GUI while demo.ts owns the loop.
export const perfToggles = {
  animate: true,
  draw: true, // SDF G-buffer draw pass (frameTick)
  voxelize: true, // scene → 3D voxel textures
  mips: true, // radiance mip pyramid
  anisoBase: true, // 6 directional level-0 volumes (iso mip0 → aniso)
  anisoMips: true, // directional-volume mip pyramids
  screenProbe: true, // screen-space probe SH (the diffuse fill/bounce source)
  cone: true, // N-cone GI gather (half-res)
  sunDepth: true, // sun shadow-map depth pass (sun-POV SDF depth)
  composite: true, // final lit image
};

// Perf folder: live per-pass toggles. Read the GPU-ms delta when a toggle flips to attribute
// cost to that pass (gpuMs comes from onSubmittedWorkDone, so it is NOT capped by vsync the
// way the rAF fps is).
export function addPerfFolder(gui: GUI) {
  const pf = gui.addFolder("Perf");
  pf.add(perfToggles, "animate").name("animate grid");
  pf.add(perfToggles, "draw").name("1· SDF draw pass");
  pf.add(perfToggles, "voxelize").name("2· voxelize");
  pf.add(perfToggles, "mips").name("3· mips");
  pf.add(perfToggles, "anisoBase").name("3a· aniso base");
  pf.add(perfToggles, "anisoMips").name("3b· aniso mips");
  pf.add(perfToggles, "screenProbe").name("4· screen probes");
  pf.add(perfToggles, "cone").name("5· cone GI");
  pf.add(perfToggles, "sunDepth").name("6· sun shadow-map pass");
  pf.add(perfToggles, "composite").name("7· composite");
  pf.open();
}
