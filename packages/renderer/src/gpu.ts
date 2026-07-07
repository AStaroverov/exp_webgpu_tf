export async function initWebGPU(
  canvas: HTMLCanvasElement,
): Promise<{ device: GPUDevice; context: GPUCanvasContext }> {
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter === null) throw new Error("No adapter found");

  // The surfel gather binds 10 storage buffers in one compute stage (posr/norw +
  // 7 scene-instance buffers + radiance cache), above the default 8. Request the
  // adapter's max (commonly 10) so it's allowed; clamps to whatever the adapter has.
  // The anisotropic-voxel base/volume passes each bind 6 storage textures in one compute
  // stage, above the default 4 → also request the adapter's max (commonly 8).
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
      maxStorageTexturesPerShaderStage: adapter.limits.maxStorageTexturesPerShaderStage,
    },
  });
  const context = canvas.getContext("webgpu") as GPUCanvasContext;

  canvas.width = canvas.clientWidth * window.devicePixelRatio;
  canvas.height = canvas.clientHeight * window.devicePixelRatio;
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format: presentationFormat,
    alphaMode: "premultiplied",
  });

  return {
    device,
    context,
  };
}
