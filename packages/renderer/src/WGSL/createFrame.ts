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
    // TEXTURE_BINDING so the Radiance Cascades pass can march it (textureLoad).
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  // --- G-buffer (MRT) written by the SDF-impostor main pass ---
  const gAlbedo = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const gNormal = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const gEmission = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  // Stopgap composite output (present()-able). Named distinctly from
  // createRCTextures' own litTexture to avoid a clash.
  const compositeTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "bgra8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  return {
    renderTexture,
    depthTexture,
    gAlbedo,
    gNormal,
    gEmission,
    compositeTexture,
    ...createRCTextures(device, canvas),
  };
}

// Radiance Cascades textures, sized rcW x rcH.
// Phase 2 (screen-space RC) marches the depth buffer directly (no JFA/distance-field):
// it consumes ONLY { cascA, cascB } (ping-pong cascades) + litTexture (the
// present()-able final image). The seed/df/emission textures below belong to the
// retired 2D-planar RC design and are kept ONLY so the off-limits engine's old
// createRadianceCascadesSystem still type-resolves; the test-scene RC ignores them.
// 0.5: depth-marching wants a bit more screen resolution than the old 2D DF path
// (RC runs at 0.25x the canvas pixels); the gather pass upsamples linearly.
export const rcDownscale = 0.5;

export function createRCTextures(device: GPUDevice, canvas: HTMLCanvasElement) {
  const rcW = Math.floor(canvas.width * rcDownscale);
  const rcH = Math.floor(canvas.height * rcDownscale);

  // --- retired 2D-planar-RC textures (engine-only; not used by screen-space RC) ---
  const emissionTexture = device.createTexture({
    size: [rcW, rcH, 1],
    format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const emitDirTexture = device.createTexture({
    size: [rcW, rcH, 1],
    format: "rg16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const seedA = device.createTexture({
    size: [rcW, rcH, 1],
    format: "rg16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const seedB = device.createTexture({
    size: [rcW, rcH, 1],
    format: "rg16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const dfTexture = device.createTexture({
    size: [rcW, rcH, 1],
    format: "r16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  // --- screen-space RC ping-pong cascades (RGB = radiance, A = resolved flag).
  // The merge walks cascade n+1 -> n reading one and writing the other, so two suffice.
  const cascA = device.createTexture({
    size: [rcW, rcH, 1],
    format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  const cascB = device.createTexture({
    size: [rcW, rcH, 1],
    format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  const litTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "bgra8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  return { emissionTexture, emitDirTexture, seedA, seedB, dfTexture, cascA, cascB, litTexture };
}

export function createFrameTick(
  {
    depthTexture,
    gAlbedo,
    gNormal,
    gEmission,
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

  // Background color for the albedo G-buffer, forced to alpha 0 so the stopgap
  // composite discards uncovered background pixels (albedo.a == 0).
  const bg = background as number[];
  const albedoClear: GPUColor = [bg[0], bg[1], bg[2], 0];

  const mainArg = {
    delta: 0,
    device,
    passEncoder: null as unknown as GPURenderPassEncoder,
  };

  const renderFrame = (commandEncoder: GPUCommandEncoder, delta: number) => {
    resizeSystem();

    // === Main Render Pass (MRT G-buffer: albedo + world normal + emission) ===
    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: gAlbedo.createView(),
          clearValue: albedoClear,
          loadOp: "clear",
          storeOp: "store",
        },
        {
          view: gNormal.createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: "clear",
          storeOp: "store",
        },
        {
          view: gEmission.createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 0, // reverse-Z, keep
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    mainArg.delta = delta;
    mainArg.passEncoder = passEncoder;
    mainCallback(mainArg);
    passEncoder.end();

    return { gAlbedo, gNormal, gEmission, depthTexture };
  };

  return renderFrame;
}
