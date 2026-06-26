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

  // Stage-3b G-buffer: per-pixel self-emission written by fs_main as a 3rd color
  // attachment (uColor.rgb * abs(material.x); a = 1). Sampled by the RC composite
  // so emitter glow is a surface property — no voxel cross-contamination / flicker.
  const emissionTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  return { renderTexture, depthTexture, normalTexture, emissionTexture };
}

export function createFrameTick(
  {
    renderTexture,
    depthTexture,
    normalTexture,
    emissionTexture,
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

  // The frame textures are fixed for this closure's lifetime (a canvas resize rebuilds the
  // whole tick via createFrameTick), so their attachment views are created ONCE here rather
  // than per frame — a per-frame createView() is pure GC churn on a stable texture.
  // Depth convention: reverse-Z. The draw pipeline uses depthCompare "greater-equal" with
  // depthClearValue 0 (NEAR maps to 1, FAR to 0).
  const depthView = depthTexture.createView();
  const textureView = renderTexture.createView();
  const normalView = normalTexture.createView();
  const emissionView = emissionTexture.createView();

  const renderFrame = (commandEncoder: GPUCommandEncoder, delta: number) => {
    resizeSystem();

    // === Main Render Pass ===
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
        {
          // G-buffer per-pixel self-emission. Cleared to 0 (background = no glow).
          view: emissionView,
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

    return { renderTexture, depthTexture, normalTexture, emissionTexture };
  };

  return renderFrame;
}
