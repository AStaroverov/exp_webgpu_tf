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

  return { renderTexture, depthTexture };
}

export function createFrameTick(
  {
    renderTexture,
    depthTexture,
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
