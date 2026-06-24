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
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // Shadow map texture - stores Z height of shadow casters (r32float for max Z value)
  const shadowMapTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "r32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  return { renderTexture, depthTexture, shadowMapTexture, ...createRCTextures(device, canvas) };
}

// Radiance Cascades textures, sized rcW x rcH.
// litTexture is canvas-sized (composite output fed to Pixelate).
// 0.4: RC runs at 0.16x the canvas pixels (and one cascade fewer than 1.0); composite upsamples linearly.
export const rcDownscale = 0.4;

export function createRCTextures(device: GPUDevice, canvas: HTMLCanvasElement) {
  const rcW = Math.floor(canvas.width * rcDownscale);
  const rcH = Math.floor(canvas.height * rcDownscale);

  const emissionTexture = device.createTexture({
    size: [rcW, rcH, 1],
    format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  // Per-emitter facing direction (normalized, Y-flipped) for directional cones.
  // (0,0) means omni; written by the emit pass as a second color attachment.
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
    renderTexture,
    depthTexture,
    shadowMapTexture,
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
  shadowMapCallback: (options: {
    delta: number;
    device: GPUDevice;
    passEncoder: GPURenderPassEncoder;
  }) => void,
) {
  const resizeSystem = createResizeSystem(canvas, getPixelRatio);

  const shadowMapArg = {
    delta: 0,
    device,
    passEncoder: null as unknown as GPURenderPassEncoder,
  };

  const mainArg = {
    delta: 0,
    device,
    passEncoder: null as unknown as GPURenderPassEncoder,
  };

  const renderFrame = (commandEncoder: GPUCommandEncoder, delta: number) => {
    resizeSystem();

    // === Shadow Map Pass (separate command buffer for synchronization) ===
    const shadowMapEncoder = device.createCommandEncoder();
    const shadowMapView = shadowMapTexture.createView();
    const shadowMapPassEncoder = shadowMapEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: shadowMapView,
          clearValue: [0, 0, 0, 0], // Z = 0 means no shadow
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    shadowMapArg.delta = delta;
    shadowMapArg.passEncoder = shadowMapPassEncoder;
    shadowMapCallback(shadowMapArg);
    shadowMapPassEncoder.end();
    // Submit shadow map pass first (creates synchronization barrier)
    device.queue.submit([shadowMapEncoder.finish()]);

    // === Main Render Pass ===
    const depthView = depthTexture.createView();
    const textureView = renderTexture.createView();

    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: background,
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

    return { renderTexture, depthTexture };
  };

  return renderFrame;
}
