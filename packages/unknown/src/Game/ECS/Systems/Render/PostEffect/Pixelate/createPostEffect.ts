import { GPUShader } from "../../../../../../../../renderer/src/WGSL/GPUShader.ts";
import { shaderMeta } from "./shader.ts";

function createSampler(device: GPUDevice) {
  return device.createSampler({
    magFilter: "nearest",
    minFilter: "nearest",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });
}

/**
 * Optional pixelate pass: sourceTexture -> own outputTexture (texture-to-texture).
 * Skipping `run` is always safe — the screen blit owns the swapchain.
 */
export function createPixelatePass(device: GPUDevice, sourceTexture: GPUTexture) {
  const outputTexture = device.createTexture({
    size: [sourceTexture.width, sourceTexture.height, 1],
    format: "bgra8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  shaderMeta.uniforms.inputSampler.setSampler(createSampler(device));
  shaderMeta.uniforms.inputTexture.setTexture(sourceTexture);
  const gpuShader = new GPUShader(shaderMeta);
  const pipeline = gpuShader.getRenderPipeline(device, "vs_main", "fs_main");
  const bindGroup = gpuShader.getBindGroup(device, 0);

  function run(commandEncoder: GPUCommandEncoder) {
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: outputTexture.createView(),
          clearValue: [0.0, 0.0, 0.0, 0.0],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(pipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(6, 1, 0, 0);
    renderPass.end();
  }

  return { run, outputTexture };
}
