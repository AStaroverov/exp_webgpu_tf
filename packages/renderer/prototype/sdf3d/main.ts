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
const KIND_CYLINDER = 2;
const KIND_RHOMBUS = 3;
const KIND_PARALLELOGRAM = 4;
const KIND_TRAPEZOID = 5;
const KIND_TRIANGLE = 6;

type Instance = {
  x: number;
  y: number;
  baseZ: number; // bottom of the object ("положение Z")
  height: number; // honest vertical extent
  hx: number; // footprint XY bounding-box half-extent X (radius for spheres)
  hy: number; // footprint XY bounding-box half-extent Y
  values: [number, number, number, number, number, number]; // shape-specific params
  roundness: number;
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
  roundness = 0,
): Instance {
  return {
    x,
    y,
    baseZ,
    height,
    hx,
    hy,
    yaw,
    kind: KIND_BOX,
    color,
    values: [hx, hy, 0, 0, 0, 0],
    roundness,
  };
}

function sphere(
  x: number,
  y: number,
  baseZ: number,
  r: number,
  color: [number, number, number],
): Instance {
  return {
    x,
    y,
    baseZ,
    height: r * 2,
    hx: r,
    hy: r,
    yaw: 0,
    kind: KIND_SPHERE,
    color,
    values: [0, 0, 0, 0, 0, 0],
    roundness: 0,
  };
}

function cylinder(
  x: number,
  y: number,
  baseZ: number,
  r: number,
  height: number,
  color: [number, number, number],
  yaw = 0,
  roundness = 0,
): Instance {
  return {
    x,
    y,
    baseZ,
    height,
    hx: r,
    hy: r,
    yaw,
    kind: KIND_CYLINDER,
    color,
    values: [r, 0, 0, 0, 0, 0],
    roundness,
  };
}

function rhombus(
  x: number,
  y: number,
  baseZ: number,
  hx: number,
  hy: number,
  height: number,
  color: [number, number, number],
  yaw = 0,
  roundness = 0,
): Instance {
  return {
    x,
    y,
    baseZ,
    height,
    hx,
    hy,
    yaw,
    kind: KIND_RHOMBUS,
    color,
    values: [hx, hy, 0, 0, 0, 0],
    roundness,
  };
}

function parallelogram(
  x: number,
  y: number,
  baseZ: number,
  hx: number,
  hy: number,
  height: number,
  color: [number, number, number],
  yaw = 0,
  skewParam = 0,
  roundness = 0,
): Instance {
  // Footprint bounding box widens by the skew amount on each side.
  return {
    x,
    y,
    baseZ,
    height,
    hx: hx + Math.abs(skewParam),
    hy,
    yaw,
    kind: KIND_PARALLELOGRAM,
    color,
    values: [hx, hy, skewParam, 0, 0, 0],
    roundness,
  };
}

function trapezoid(
  x: number,
  y: number,
  baseZ: number,
  r1: number,
  r2: number,
  he: number,
  height: number,
  color: [number, number, number],
  yaw = 0,
  roundness = 0,
): Instance {
  return {
    x,
    y,
    baseZ,
    height,
    hx: Math.max(r1, r2),
    hy: he,
    yaw,
    kind: KIND_TRAPEZOID,
    color,
    values: [r1, r2, he, 0, 0, 0],
    roundness,
  };
}

function triangle(
  x: number,
  y: number,
  baseZ: number,
  hx: number,
  hy: number,
  height: number,
  color: [number, number, number],
  yaw = 0,
  roundness = 0,
): Instance {
  return {
    x,
    y,
    baseZ,
    height,
    hx,
    hy,
    yaw,
    kind: KIND_TRIANGLE,
    color,
    values: [hx, hy, 0, 0, 0, 0],
    roundness,
  };
}

// --- Scene: a ground slab + at least one of every kind, including a platform
// with a box standing ON it so "положение Z" (base elevation) stays visually
// distinct from honest object height. A couple of shapes vary yaw/roundness.
const scene: Instance[] = [
  box(0, 0, -0.4, 26, 26, 0.4, [0.16, 0.18, 0.22]), // ground (extruded box)

  // Spheres (true 3D)
  sphere(-6, 2, 0, 2.2, [0.9, 0.8, 0.35]),
  sphere(0, 3, 0, 1.4, [0.8, 0.4, 0.6]),
  sphere(6, 1, 0, 1.8, [0.5, 0.85, 0.8]),

  // Extruded 2D footprints — one of every remaining kind.
  box(-9, -6, 0, 2, 2, 6, [0.85, 0.45, 0.3]), // kind 1: tall tower
  cylinder(-3, -7, 0, 2.5, 2.5, [0.4, 0.7, 0.9]), // kind 2: cylinder
  rhombus(4, -6, 0, 1.8, 2.5, 3, [0.6, 0.8, 0.4], 0.5), // kind 3: rotated rhombus
  parallelogram(-9, 6, 0, 1.5, 2.5, 2.5, [0.7, 0.5, 0.85], 0.0, 1.2), // kind 4
  trapezoid(-3, 7, 0, 2.5, 1.0, 2.0, 2.5, [0.85, 0.6, 0.3], 0.0, 0.3), // kind 5 (rounded)
  triangle(3, 7, 0, 2.2, 2.2, 2.5, [0.55, 0.85, 0.9], 0.6), // kind 6: rotated triangle

  // Platform + a box standing ON the platform (baseZ = platform top) + sphere on top.
  box(9, 7, 0, 4, 4, 1.5, [0.3, 0.32, 0.4]),
  box(9, 7, 1.5, 1.4, 1.4, 3, [0.95, 0.55, 0.55], 0, 0.4), // rounded box
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
  // Inst = 5 x vec4 = 20 floats / 80 bytes (std140 storage):
  //   centerYaw, halfKindR, values01h, values234, color.
  const data = new Float32Array(list.length * 20);
  for (let i = 0; i < list.length; i++) {
    const o = list[i];
    const hz = o.height / 2;
    const base = i * 20;
    // centerYaw
    data[base + 0] = o.x;
    data[base + 1] = o.y;
    data[base + 2] = o.baseZ + hz; // store center Z
    data[base + 3] = o.yaw;
    // halfKindR: hx, hy, roundness, kind
    data[base + 4] = o.hx;
    data[base + 5] = o.hy;
    data[base + 6] = o.roundness;
    data[base + 7] = o.kind;
    // values01h: values[0], values[1], height, pad
    data[base + 8] = o.values[0];
    data[base + 9] = o.values[1];
    data[base + 10] = o.height;
    data[base + 11] = 0;
    // values234: values[2], values[3], values[4], values[5]
    data[base + 12] = o.values[2];
    data[base + 13] = o.values[3];
    data[base + 14] = o.values[4];
    data[base + 15] = o.values[5];
    // color
    data[base + 16] = o.color[0];
    data[base + 17] = o.color[1];
    data[base + 18] = o.color[2];
    data[base + 19] = 0;
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
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "read-only-storage" },
      },
    ],
  });
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module,
      entryPoint: "vs",
      buffers: [
        { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] },
      ],
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
  const elevation = (70 * Math.PI) / 180; // angle above the ground plane (90 = straight down)
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
