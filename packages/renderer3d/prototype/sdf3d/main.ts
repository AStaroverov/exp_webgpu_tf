// 2.5D SDF impostor prototype — standalone, raw WebGPU.
//
// Demonstrates the proposed renderer direction:
//   - orthographic camera tilted off vertical (top-down 2.5D),
//   - sphere + box primitives as true 3D SDFs (honest volume & height),
//   - "положение Z" (base elevation) and honest object height as separate values,
//   - real depth from the ray hit → correct occlusion/sorting, no CPU sort,
//   - one directional light for plausible shading.
//
// Run: `npm run dev` inside packages/renderer, open the served page.

import { mat4, vec3 } from "gl-matrix";
import { initWebGPU } from "../../src/gpu.ts";
import { shaderCode } from "./shader.ts";

const KIND_SPHERE = 0;
const KIND_BOX = 1;

type Instance = {
  x: number;
  y: number;
  baseZ: number; // bottom of the object ("положение Z")
  height: number; // honest vertical extent
  hx: number; // half-extent X (radius for spheres)
  hy: number; // half-extent Y
  yaw: number;
  kind: number;
  color: [number, number, number];
};

function box(
  x: number,
  y: number,
  baseZ: number,
  hx: number,
  hy: number,
  height: number,
  color: [number, number, number],
  yaw = 0,
): Instance {
  return { x, y, baseZ, height, hx, hy, yaw, kind: KIND_BOX, color };
}

function sphere(
  x: number,
  y: number,
  baseZ: number,
  r: number,
  color: [number, number, number],
): Instance {
  return { x, y, baseZ, height: r * 2, hx: r, hy: r, yaw: 0, kind: KIND_SPHERE, color };
}

// --- Scene: a ground slab + a mix of boxes/spheres, including one box raised on
// a platform so "положение Z" (base elevation) is visually distinct from height.
const scene: Instance[] = [
  box(0, 0, -0.4, 26, 26, 0.4, [0.16, 0.18, 0.22]), // ground

  box(-9, -6, 0, 2, 2, 6, [0.85, 0.45, 0.3]), // tall tower
  box(-3, -7, 0, 2.5, 2.5, 2, [0.4, 0.7, 0.9]), // low wide block
  box(4, -6, 0, 1.5, 4, 3, [0.6, 0.8, 0.4], 0.5), // rotated block

  sphere(-6, 2, 0, 2.2, [0.9, 0.8, 0.35]),
  sphere(0, 3, 0, 1.4, [0.8, 0.4, 0.6]),
  sphere(6, 1, 0, 1.8, [0.5, 0.85, 0.8]),

  // platform + a box standing ON the platform (baseZ = platform top)
  box(9, 7, 0, 4, 4, 1.5, [0.3, 0.32, 0.4]),
  box(9, 7, 1.5, 1.4, 1.4, 3, [0.95, 0.55, 0.55]),
  sphere(9, 7, 4.5, 1, [0.95, 0.95, 0.95]),
];

// 36-vertex unit cube ([-1,1]^3), 12 triangles.
function unitCube(): Float32Array {
  const c = [
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1],
  ];
  const faces = [
    [0, 1, 2, 0, 2, 3], // -z
    [4, 6, 5, 4, 7, 6], // +z
    [0, 4, 5, 0, 5, 1], // -y
    [3, 2, 6, 3, 6, 7], // +y
    [0, 3, 7, 0, 7, 4], // -x
    [1, 5, 6, 1, 6, 2], // +x
  ];
  const out: number[] = [];
  for (const f of faces) for (const i of f) out.push(...c[i]);
  return new Float32Array(out);
}

function packInstances(list: Instance[]): Float32Array {
  // Inst = vec4 centerYaw, vec4 halfKind, vec4 color = 12 floats.
  const data = new Float32Array(list.length * 12);
  for (let i = 0; i < list.length; i++) {
    const o = list[i];
    const hz = o.height / 2;
    const base = i * 12;
    data[base + 0] = o.x;
    data[base + 1] = o.y;
    data[base + 2] = o.baseZ + hz; // store center Z
    data[base + 3] = o.yaw;
    data[base + 4] = o.hx;
    data[base + 5] = o.hy;
    data[base + 6] = hz;
    data[base + 7] = o.kind;
    data[base + 8] = o.color[0];
    data[base + 9] = o.color[1];
    data[base + 10] = o.color[2];
    data[base + 11] = 0;
  }
  return data;
}

async function main() {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const { device, context } = await initWebGPU(canvas);
  const format = navigator.gpu.getPreferredCanvasFormat();

  const cubeData = unitCube();
  const cubeBuffer = device.createBuffer({
    size: cubeData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(cubeBuffer, 0, cubeData);

  const instData = packInstances(scene);
  const instBuffer = device.createBuffer({
    size: instData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(instBuffer, 0, instData);

  // Uniforms: mat4 viewProj (64) + vec4 rayDir (16) + vec4 lightDir (16) = 96 bytes.
  const uniformData = new Float32Array(24);
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const module = device.createShaderModule({ code: shaderCode });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
    ],
  });
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module,
      entryPoint: "vs",
      buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }],
    },
    fragment: { module, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: { format: "depth32float", depthCompare: "less", depthWriteEnabled: true },
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: instBuffer } },
    ],
  });

  let depthTexture: GPUTexture | null = null;
  function ensureDepth(w: number, h: number) {
    if (depthTexture && depthTexture.width === w && depthTexture.height === h) return depthTexture;
    depthTexture?.destroy();
    depthTexture = device.createTexture({
      size: [w, h, 1],
      format: "depth32float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    return depthTexture;
  }

  // Camera controls.
  let azimuth = Math.PI * 0.25; // orbit angle around Z
  const elevation = (52 * Math.PI) / 180; // angle above the ground plane (90 = straight down)
  let zoom = 26; // world half-height visible
  let orbit = true;
  const target = vec3.fromValues(0, 0, 1.5);

  canvas.addEventListener("mousedown", () => (dragging = true));
  window.addEventListener("mouseup", () => (dragging = false));
  let dragging = false;
  window.addEventListener("mousemove", (e) => {
    if (dragging) azimuth -= e.movementX * 0.01;
  });
  window.addEventListener(
    "wheel",
    (e) => {
      zoom = Math.max(8, Math.min(60, zoom * (1 + Math.sign(e.deltaY) * 0.1)));
    },
    { passive: true },
  );
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") orbit = !orbit;
  });

  const view = mat4.create();
  const proj = mat4.create();
  const viewProj = mat4.create();
  // Remap gl-matrix OpenGL depth ([-1,1]) to WebGPU ([0,1]).
  const ndcFix = mat4.fromValues(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 1);
  const eye = vec3.create();
  const lightDir = vec3.normalize(vec3.create(), vec3.fromValues(-0.4, -0.55, -0.72));

  let last = performance.now();
  function frame(now: number) {
    const dt = (now - last) / 1000;
    last = now;
    if (orbit) azimuth += dt * 0.3;

    const w = (canvas.width = Math.floor(canvas.clientWidth * devicePixelRatio));
    const h = (canvas.height = Math.floor(canvas.clientHeight * devicePixelRatio));
    const aspect = w / h;

    // Orthographic tilted top-down camera.
    const ce = Math.cos(elevation);
    const se = Math.sin(elevation);
    const dist = 100;
    const dir = vec3.fromValues(ce * Math.cos(azimuth), ce * Math.sin(azimuth), se);
    vec3.scaleAndAdd(eye, target, dir, dist);
    mat4.lookAt(view, eye, target, vec3.fromValues(0, 0, 1));
    mat4.ortho(proj, -zoom * aspect, zoom * aspect, -zoom, zoom, 0.1, dist + 200);
    mat4.multiply(proj, ndcFix, proj);
    mat4.multiply(viewProj, proj, view);

    const rayDir = vec3.normalize(vec3.create(), vec3.subtract(vec3.create(), target, eye));

    uniformData.set(viewProj, 0);
    uniformData.set([rayDir[0], rayDir[1], rayDir[2], 0], 16);
    uniformData.set([lightDir[0], lightDir[1], lightDir[2], 0], 20);
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const depth = ensureDepth(w, h);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.043, g: 0.051, b: 0.07, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: depth.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, cubeBuffer);
    pass.draw(36, scene.length, 0, 0);
    pass.end();
    device.queue.submit([encoder.finish()]);

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="color:#f88;padding:20px">${err?.stack ?? err}</pre>`;
  console.error(err);
});
