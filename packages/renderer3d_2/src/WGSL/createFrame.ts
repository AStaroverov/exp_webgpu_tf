import { createResizeSystem } from "../ECS/Systems/ResizeSystem.ts";

export function createFrameTextures(device: GPUDevice, canvas: HTMLCanvasElement) {
  const renderTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "bgra8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  const depthTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "depth32float",
    // TEXTURE_BINDING so the world-RC composite can sample reverse-Z depth
    // (textureLoad on a texture_depth_2d) to reconstruct world position. The
    // main pass still uses it as a render attachment; behavior is unchanged.
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  // Stage-3b G-buffer: world-space normals written by fs_main as a 2nd color
  // attachment (packed *0.5+0.5; a = surface mask). Sampled by the RC composite
  // for the normal-aware directional term.
  const normalTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  return { renderTexture, depthTexture, normalTexture };
}

// World-space RC probe-grid constants (Stage 3 — height layers, Model A). Per
// cascade c the atlas is [(gridX>>c)*(dir0W<<c), (gridY>>c)*(dir0W<<c)] with gridZ
// array layers (one horizontal probe sheet per layer). Mirror these in the system
// params. With gridX=gridY=128, dir0W=4 the >>c / <<c cancel so each cascade atlas is
// 512x512 — but x and y can now differ if gridX!=gridY.
export const WORLD_GRID_X = 128; // probes along x, cascade 0
export const WORLD_GRID_Y = 128; // probes along y, cascade 0
export const WORLD_GRID_Z = 6; // number of horizontal probe sheets (array layers)
export const WORLD_DIR0_W = 4; // octahedral tile side -> 16 directions per probe
// Number of cascades (Stage 2). Per cascade c: probes/side /= 2, dir-tile side *= 2,
// so the atlas side (GRID_*_c * DIR_W_c) stays CONSTANT for every cascade. cell_c =
// cell0*2^c, interval reach ~ baseInterval*4^(N-1). gridX/gridY must be divisible by
// 2^(N-1): 128 / 2^4 = 8 ok. gridZ/cellZ are CONSTANT across cascades (Model A).
export const WORLD_CASCADE_COUNT = 5;

export type RCTextureDims = {
  gridX: number;
  gridY: number;
  gridZ: number;
  dir0W: number;
};

export const DEFAULT_RC_TEXTURE_DIMS: RCTextureDims = {
  gridX: WORLD_GRID_X,
  gridY: WORLD_GRID_Y,
  gridZ: WORLD_GRID_Z,
  dir0W: WORLD_DIR0_W,
};

export function createRCTextures(
  device: GPUDevice,
  canvas: HTMLCanvasElement,
  dims: RCTextureDims = DEFAULT_RC_TEXTURE_DIMS,
) {
  // === World-space RC (Stage 3) — cascade hierarchy + height layers. ===
  // probe (i,j) of cascade c owns a DIR_W_c x DIR_W_c octahedral tile; the atlas is a
  // 2D-ARRAY texture with gridZ layers, layer k = the horizontal probe sheet at world
  // height z_k. rgb = interval radiance, a = visibility (1 = ray passed the whole
  // interval unobstructed). COPY_SRC so diagnostics can read back a chosen layer.
  const { gridX, gridY, gridZ, dir0W } = dims;
  const mkAtlas = (c: number) => {
    const w = Math.max(1, gridX >> c) * (dir0W << c);
    const h = Math.max(1, gridY >> c) * (dir0W << c);
    return device.createTexture({
      // 2D-array: [w, h, gridZ]. dimension "2d" is correct for a 2D-array texture.
      size: [w, h, gridZ],
      dimension: "2d",
      format: "rgba16float",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });
  };
  // probeRad[c] = raw gather for cascade c (c in 0..N-1).
  const probeRad = Array.from({ length: WORLD_CASCADE_COUNT }, (_, c) => mkAtlas(c));
  // probeMerge[c] = cascade c after merging c+1 into it (c in 0..N-2). The top
  // cascade needs no merge target (its merged value IS probeRad[N-1]).
  const probeMerge = Array.from({ length: WORLD_CASCADE_COUNT - 1 }, (_, c) => mkAtlas(c));

  // Canvas-sized composite output — the presented texture.
  const worldLitTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "bgra8unorm",
    // COPY_SRC so the diagnostics can read the final composite back to the CPU.
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC,
  });

  return { probeRad, probeMerge, worldLitTexture };
}

export function createFrameTick(
  {
    renderTexture,
    depthTexture,
    normalTexture,
    canvas,
    device,
    background,
    getPixelRatio,
  }: ReturnType<typeof createFrameTextures> & {
    canvas: HTMLCanvasElement;
    device: GPUDevice;
    background: GPUColor;
    getPixelRatio: () => number;
  },
  mainCallback: (options: {
    delta: number;
    device: GPUDevice;
    passEncoder: GPURenderPassEncoder;
  }) => void,
) {
  const resizeSystem = createResizeSystem(canvas, getPixelRatio);

  const mainArg = {
    delta: 0,
    device,
    passEncoder: null as unknown as GPURenderPassEncoder,
  };

  const renderFrame = (commandEncoder: GPUCommandEncoder, delta: number) => {
    resizeSystem();

    // === Main Render Pass ===
    // Depth convention: reverse-Z. The draw pipeline uses depthCompare
    // "greater-equal" with depthClearValue 0 (NEAR maps to 1, FAR to 0).
    const depthView = depthTexture.createView();
    const textureView = renderTexture.createView();
    const normalView = normalTexture.createView();

    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: background,
          loadOp: "clear",
          storeOp: "store",
        },
        {
          // G-buffer world normals. Cleared to 0 → a = 0 = "no surface".
          view: normalView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    mainArg.delta = delta;
    mainArg.passEncoder = passEncoder;
    mainCallback(mainArg);
    passEncoder.end();

    return { renderTexture, depthTexture, normalTexture };
  };

  return renderFrame;
}
