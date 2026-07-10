// GPU per-stage frame profiler via WebGPU TIMESTAMP QUERIES. Module-level singleton (the
// SunLight/viewProjMatrix pattern) so any pass site can tag itself without threading a handle
// through every subsystem's deps.
//
// How it works (per the spec — timestamps exist ONLY at pass boundaries; encoder.writeTimestamp
// was removed from WebGPU in 2023):
//   1. gpu.ts requests the OPTIONAL "timestamp-query" feature when the adapter has it.
//   2. Each begin{Compute,Render}Pass site passes `timestampWrites: gpuSpan("label")` — a pair of
//      slots in one shared GPUQuerySet (begin/end of that pass). Multi-pass stages (mips, aniso,
//      voxelize) reuse ONE label; the readback SUMS same-label spans into a single row.
//   3. End of frame: resolveQuerySet → a QUERY_RESOLVE buffer, then (when the staging buffer is
//      not mapped) copy to a MAP_READ staging.
//   4. After submit: mapAsync → BigUint64Array of NANOSECONDS → per-label ms, EMA-smoothed.
//
// When the feature is unavailable (or init was never called — e.g. the headless engine path),
// gpuSpan returns undefined, which pass descriptors accept → zero overhead, no branching at sites.
//
// Chrome quantizes timestamps to 100 µs by default (fingerprinting mitigation) — fine for ms-scale
// passes; launch with --enable-webgpu-developer-features for µs precision. Some GPUs can return
// end < begin across command-buffer boundaries — negative deltas are clamped to 0.

const MAX_SPANS = 64; // spans per frame (a span = one pass); 2 query slots each

let device: GPUDevice | null = null;
let querySet: GPUQuerySet | null = null;
let resolveBuf: GPUBuffer | null = null;
let stagingBuf: GPUBuffer | null = null;

let spanCount = 0;
const spanLabels: string[] = new Array(MAX_SPANS);
// Labels of the frame whose timestamps are sitting in stagingBuf awaiting mapAsync (null = none).
let pendingLabels: string[] | null = null;
let mapInFlight = false;

// EMA-smoothed per-label durations, ms. Read by the GUI every frame.
const emaMs = new Map<string, number>();
const EMA = 0.2;

// Create the query/readback resources. Call once after device creation; returns false (and leaves
// the module inert) when the adapter/device lacks "timestamp-query".
export function initGpuTimer(dev: GPUDevice): boolean {
  if (!dev.features.has("timestamp-query")) return false;
  device = dev;
  querySet = dev.createQuerySet({ type: "timestamp", count: MAX_SPANS * 2 });
  resolveBuf = dev.createBuffer({
    size: MAX_SPANS * 2 * 8, // one u64 per slot
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  stagingBuf = dev.createBuffer({
    size: MAX_SPANS * 2 * 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  return true;
}

// Reset the per-frame span allocator. Call at the head of the frame, before any pass is encoded.
export function gpuTimerBeginFrame(): void {
  spanCount = 0;
}

// Allocate a begin/end slot pair for one pass. The returned dictionary goes straight into the
// pass descriptor's `timestampWrites`. Shared shape between compute and render passes.
export function gpuSpan(label: string): GPUComputePassTimestampWrites | undefined {
  if (querySet === null || spanCount >= MAX_SPANS) return undefined;
  const i = spanCount++;
  spanLabels[i] = label;
  return {
    querySet,
    beginningOfPassWriteIndex: i * 2,
    endOfPassWriteIndex: i * 2 + 1,
  };
}

// Encode the readback: resolve this frame's slots + (if the staging buffer is free) copy them out
// and snapshot the labels. Call ONCE at the end of the frame, on the same encoder, after every
// instrumented pass.
export function gpuTimerResolve(encoder: GPUCommandEncoder): void {
  if (querySet === null || spanCount === 0) return;
  encoder.resolveQuerySet(querySet, 0, spanCount * 2, resolveBuf!, 0);
  if (!mapInFlight && pendingLabels === null) {
    encoder.copyBufferToBuffer(resolveBuf!, 0, stagingBuf!, 0, spanCount * 2 * 8);
    pendingLabels = spanLabels.slice(0, spanCount);
  }
}

// Kick the async map of the staging copy (call right after queue.submit). Self-throttling: while
// a map is in flight the per-frame copy above is skipped, so the buffer is never touched mapped.
export function gpuTimerPoll(): void {
  if (device == null || mapInFlight || pendingLabels == null || stagingBuf == null) return;
  const frameLabels = pendingLabels;
  mapInFlight = true;
  void stagingBuf
    .mapAsync(GPUMapMode.READ, 0, frameLabels.length * 16)
    .then(() => {
      const words = new BigUint64Array(
        stagingBuf!.getMappedRange(0, frameLabels.length * 16).slice(0),
      );
      stagingBuf!.unmap();
      // ADDITIVE attribution via END-timestamp deltas — NOT the raw [begin, end] windows. Raw
      // windows OVERLAP on real GPUs (the begin stamp is top-of-pipe: it fires while the previous
      // pass's fragment work still drains), so window sums exceeded the real frame ~3× (measured).
      // End stamps are bottom-of-pipe completion times and monotonic along the queue, so
      //   dur_i = end_i − max(end_{i−1}, begin_i)
      // is each pass's EXCLUSIVE tail: exact for hazard-serialized passes (our GI chain — every
      // pass reads its predecessor's output), and overlapped time is attributed to the LATER pass
      // for independent ones (draw ∥ sunDepth). Idle gaps between passes are left unattributed,
      // so Σ rows ≤ frameSpan. Spans are indexed in ENCODE order == queue execution order.
      const sums = new Map<string, number>();
      let prevEnd = words[0]; // first pass: dur = end − begin (nothing precedes it)
      for (let i = 0; i < frameLabels.length; i++) {
        const begin = words[i * 2];
        const end = words[i * 2 + 1];
        const from = begin > prevEnd ? begin : prevEnd;
        const ms = Math.max(0, Number(end - from)) / 1e6;
        sums.set(frameLabels[i], (sums.get(frameLabels[i]) ?? 0) + ms);
        if (end > prevEnd) prevEnd = end;
      }
      // frameSpan = first begin → last end: the frame's GPU time as ONE window (compare against
      // the wall-clock onSubmittedWorkDone number; Σ of the rows above ≈ this, minus idle gaps).
      let tMin = words[0];
      let tMax = words[1];
      for (let i = 1; i < frameLabels.length; i++) {
        if (words[i * 2] < tMin) tMin = words[i * 2];
        if (words[i * 2 + 1] > tMax) tMax = words[i * 2 + 1];
      }
      sums.set("frameSpan", Math.max(0, Number(tMax - tMin)) / 1e6);
      for (const [label, ms] of sums) {
        const prev = emaMs.get(label);
        emaMs.set(label, prev === undefined ? ms : prev * (1 - EMA) + ms * EMA);
      }
    })
    .finally(() => {
      mapInFlight = false;
      pendingLabels = null;
    });
}

// EMA-smoothed per-label durations in ms (empty until the first readback lands).
export function getGpuTimings(): ReadonlyMap<string, number> {
  return emaMs;
}
