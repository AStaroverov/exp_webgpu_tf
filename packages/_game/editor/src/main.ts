import { hasComponent } from "bitecs";
import { fromEvent, map, Subject, Subscription, switchMap, takeUntil } from "rxjs";
import { createEngine } from "../../../engine/src/createEngine.ts";
import {
  createEntityId,
  type EngineComponents,
  type EngineWorld,
  getEngineComponents,
} from "../../../engine/src/ECS/createEngineWorld.ts";
import { removeEntityTree } from "../../../engine/src/ECS/hierarchy.ts";
import { ShapeKind } from "../../../renderer/src/ECS/Components/Shape.ts";
import { createRectangle } from "../../../renderer/src/ECS/Entities/Shapes.ts";
import { SunLight } from "../../../renderer/src/ECS/Systems/SunLight.ts";
import {
  cameraAzimuth,
  cameraElevation,
  cameraHeight,
  cameraZoom,
  setCameraAzimuth,
  setCameraElevation,
  setCameraPosition,
  setCameraZoom,
} from "../../../renderer/src/ECS/Systems/ResizeSystem.ts";
import { ENTITIES, type EntityAnimations, type EntityInstance } from "./Entities/registry.ts";
import { clips$, makeClipAnimations, registerClip } from "./anim/registry.ts";
import { animatableBones, editToClip, snapshotPose, type EditClip } from "./anim/editclip.ts";
import { readPose, writePose, type Pose } from "./anim/pose.ts";
import {
  EDIT,
  editClip$,
  selectedAnimation$,
  selectedEid$,
  selectedEntityId$,
  selectedScale$,
} from "./state.ts";

const NONE = "none";

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const selectEl = document.getElementById("entity-select") as HTMLSelectElement;
  const animSelectEl = document.getElementById("animation-select") as HTMLSelectElement;
  const scaleEl = document.getElementById("scale-input") as HTMLInputElement;
  const regenBtn = document.getElementById("regen") as HTMLButtonElement;
  const treeEl = document.getElementById("tree") as HTMLElement;
  const componentsEl = document.getElementById("components") as HTMLElement;
  const inspectorEl = document.getElementById("inspector") as HTMLElement;
  const animPanelEl = document.getElementById("anim") as HTMLElement;
  const clipNameEl = document.getElementById("clip-name") as HTMLInputElement;
  const clipDurationEl = document.getElementById("clip-duration") as HTMLInputElement;
  const snapshotEl = document.getElementById("snapshot") as HTMLButtonElement;
  const keysEl = document.getElementById("keys") as HTMLElement;
  const logClipEl = document.getElementById("log-clip") as HTMLButtonElement;
  const animHintEl = document.getElementById("anim-hint") as HTMLElement;
  const poseFields = {
    tx: document.getElementById("pose-tx") as HTMLInputElement,
    ty: document.getElementById("pose-ty") as HTMLInputElement,
    tz: document.getElementById("pose-tz") as HTMLInputElement,
    rx: document.getElementById("pose-rx") as HTMLInputElement,
    ry: document.getElementById("pose-ry") as HTMLInputElement,
    rz: document.getElementById("pose-rz") as HTMLInputElement,
  } satisfies Record<keyof Pose, HTMLInputElement>;

  animHintEl.textContent =
    "Set Animation = edit. Select a part, edit its pose, click Snapshot to add a record. " +
    "Each record has a key + % time; same key merges into one keyframe. Set Duration, " +
    "pick the clip by name in Animation to preview, Log clip → console.";

  const engine = await createEngine({ canvas });
  const world = engine.world as EngineWorld;
  const sceneRoot = engine.sceneRoot;

  const components = getEngineComponents(world);
  const componentEntries = Object.entries(components) as Array<
    [string, EngineComponents[keyof EngineComponents]]
  >;

  const subs = new Subscription();
  const rows: HTMLElement[] = [];

  const isEdit = () => selectedAnimation$.value === EDIT;

  function nodeLabel(eid: number): string {
    const { Shape } = components;
    if (hasComponent(world, eid, Shape)) {
      const k = Shape.kind[eid];
      return `${ShapeKind[k] ?? "Shape"} #${eid}`;
    }
    return `#${eid}`;
  }

  function appendNode(eid: number, label: string, depth: number): void {
    const { Children } = components;
    const row = document.createElement("div");
    row.textContent = label;
    row.dataset.eid = String(eid);
    row.style.paddingLeft = `${depth * 14}px`;
    treeEl.append(row);
    rows.push(row);
    if (hasComponent(world, eid, Children)) {
      const count = Children.entitiesCount.get(eid);
      for (let i = 0; i < count; i++) {
        appendNode(
          Children.entitiesIds.get(eid, i),
          nodeLabel(Children.entitiesIds.get(eid, i)),
          depth + 1,
        );
      }
    }
  }

  function renderTree(rootEid: number, rootLabel: string): void {
    treeEl.replaceChildren();
    rows.length = 0;
    appendNode(rootEid, rootLabel, 0);
  }

  function applyHighlight(eid: number): void {
    for (let i = 0; i < rows.length; i++) {
      rows[i].classList.toggle("selected", Number(rows[i].dataset.eid) === eid);
    }
  }

  function renderComponents(eid: number): void {
    componentsEl.replaceChildren();
    if (eid < 0) return;
    for (let i = 0; i < componentEntries.length; i++) {
      const [name, comp] = componentEntries[i];
      if (hasComponent(world, eid, comp)) {
        const row = document.createElement("div");
        row.textContent = name;
        componentsEl.append(row);
      }
    }
  }

  const poseScratch: Pose = { tx: 0, ty: 0, tz: 0, rx: 0, ry: 0, rz: 0 };
  const poseKeys = Object.keys(poseFields) as (keyof Pose)[];

  function poseEditable(eid: number): boolean {
    return eid >= 0 && hasComponent(world, eid, components.LocalTransform);
  }

  function readInspector(eid: number): void {
    inspectorEl.classList.toggle("disabled", !(isEdit() && poseEditable(eid)));
    if (!poseEditable(eid)) return;
    readPose(components.LocalTransform.matrix.getBatch(eid), poseScratch);
    for (let i = 0; i < poseKeys.length; i++) {
      const field = poseFields[poseKeys[i]];
      if (document.activeElement === field) continue;
      field.value = poseScratch[poseKeys[i]].toFixed(3);
    }
  }

  function writeInspector(): void {
    const eid = selectedEid$.value;
    if (!isEdit() || !poseEditable(eid)) return;
    for (let i = 0; i < poseKeys.length; i++) {
      poseScratch[poseKeys[i]] = Number(poseFields[poseKeys[i]].value);
    }
    writePose(components.LocalTransform.matrix.getBatch(eid), poseScratch);
  }

  // ── Animation pipeline (right panel) ────────────────────────────────────────

  function ensureEditClip(): EditClip | null {
    if (currentInstance === null) return null;
    const edit = editClip$.value;
    if (edit !== null && edit.entityId === currentEntityId) return edit;
    return {
      entityId: currentEntityId,
      name: clipNameEl.value.trim() || "clip",
      duration: Number(clipDurationEl.value) || 2,
      bones: animatableBones(currentInstance),
      records: [],
    };
  }

  function registerEdit(edit: EditClip): void {
    if (edit.records.length === 0) return;
    registerClip(edit.entityId, { name: edit.name, loop: true, clip: editToClip(edit) });
  }

  function renderRecords(edit: EditClip | null): void {
    keysEl.replaceChildren();
    if (edit === null) return;
    for (let i = 0; i < edit.records.length; i++) {
      const row = document.createElement("div");
      const key = document.createElement("input");
      key.type = "number";
      key.step = "1";
      key.value = String(edit.records[i].key);
      key.dataset.idx = String(i);
      key.dataset.field = "key";
      const pct = document.createElement("input");
      pct.type = "number";
      pct.step = "1";
      pct.min = "0";
      pct.max = "100";
      pct.value = String(edit.records[i].pct);
      pct.dataset.idx = String(i);
      pct.dataset.field = "pct";
      const rm = document.createElement("span");
      rm.textContent = "✕";
      rm.className = "rm";
      rm.dataset.rm = String(i);
      row.append(key, pct, rm);
      keysEl.append(row);
    }
  }

  function snapshot(): void {
    const edit = ensureEditClip();
    if (edit === null || currentInstance === null) return;
    const last = edit.records[edit.records.length - 1];
    const key = last === undefined ? 0 : last.key + 1;
    const pct = edit.records.length === 0 ? 0 : 100;
    edit.records.push({ key, pct, pose: snapshotPose(world, currentInstance, edit.bones) });
    editClip$.next(edit);
  }

  function setRecordField(i: number, field: "key" | "pct", v: number): void {
    const edit = editClip$.value;
    if (edit === null || i >= edit.records.length) return;
    edit.records[i][field] = v;
    registerEdit(edit);
  }

  function removeRecord(i: number): void {
    const edit = editClip$.value;
    if (edit === null) return;
    edit.records.splice(i, 1);
    editClip$.next(edit);
  }

  function logClip(): void {
    const edit = editClip$.value;
    if (edit === null || edit.records.length === 0) {
      console.log("[anim] nothing to log — snapshot a pose first");
      return;
    }
    console.log(
      `[anim] clip "${edit.name}" for "${edit.entityId}"\n` +
        JSON.stringify({ name: edit.name, loop: true, ...editToClip(edit) }, null, 2),
    );
  }

  // ── scene + camera ──────────────────────────────────────────────────────────

  SunLight.enabled = true;
  SunLight.angle = 2.4;
  SunLight.elevation = 0.95;
  SunLight.intensity = 0.9;
  SunLight.color = [1.0, 0.93, 0.82];

  setCameraPosition(0, 0);
  cameraHeight.value = 1.5;
  cameraZoom.value = 48;

  // Point the camera (and thus the zoom pivot) at a selected entity's world position. Read after a
  // tick so the GlobalTransform is fresh — a just-built entity still has the identity matrix.
  let pendingFocus = -1;
  function focusCamera(eid: number): void {
    if (eid < 0 || !hasComponent(world, eid, components.GlobalTransform)) return;
    const m = components.GlobalTransform.matrix.getBatch(eid);
    setCameraPosition(m[12], m[13]);
    cameraHeight.value = m[14];
  }
  cameraElevation.value = 55;
  cameraAzimuth.value = 45;

  const ground = createRectangle(world, {
    x: 0,
    y: 0,
    z: -0.5,
    width: 60,
    height: 60,
    depth: 1,
    color: [0.18, 0.2, 0.24, 1],
    eid: createEntityId(world),
  });
  components.Children.addChild(sceneRoot, ground);

  for (let i = 0; i < ENTITIES.length; i++) {
    const option = document.createElement("option");
    option.value = ENTITIES[i].id;
    option.textContent = ENTITIES[i].label;
    selectEl.append(option);
  }

  let currentRoot = -1;
  let currentEntityId = ENTITIES[0].id;
  let currentInstance: EntityInstance | null = null;
  let currentAnimations: EntityAnimations = {};

  function refreshAnimations(): void {
    if (currentInstance === null) return;
    currentAnimations = {
      ...currentInstance.animations,
      ...makeClipAnimations(world, currentEntityId, currentInstance),
    };
    fillAnimationOptions();
  }

  function build(id: string): void {
    if (currentRoot >= 0) {
      components.Children.removeChild(sceneRoot, currentRoot);
      removeEntityTree(world, currentRoot);
    }
    const def = ENTITIES.find((d) => d.id === id) ?? ENTITIES[0];
    const instance = def.build(world, { scale: selectedScale$.value });
    currentRoot = instance.root;
    currentEntityId = def.id;
    currentInstance = instance;
    components.Children.addChild(sceneRoot, currentRoot);
    if (editClip$.value !== null && editClip$.value.entityId !== def.id) editClip$.next(null);
    refreshAnimations();
    renderTree(currentRoot, def.label);
    selectedEid$.next(currentRoot);
  }

  function fillAnimationOptions(): void {
    const names = [NONE, EDIT, ...Object.keys(currentAnimations)];
    animSelectEl.replaceChildren();
    for (let i = 0; i < names.length; i++) {
      const option = document.createElement("option");
      option.value = names[i];
      option.textContent = names[i];
      animSelectEl.append(option);
    }
    animSelectEl.value = names.includes(selectedAnimation$.value) ? selectedAnimation$.value : NONE;
  }

  const rebuild$ = new Subject<void>();

  subs.add(selectedEntityId$.subscribe((id) => build(id)));
  subs.add(rebuild$.subscribe(() => build(selectedEntityId$.value)));
  subs.add(
    selectedEid$.subscribe((eid) => {
      applyHighlight(eid);
      renderComponents(eid);
      readInspector(eid);
      pendingFocus = eid;
    }),
  );

  subs.add(
    selectedAnimation$.subscribe((a) => {
      if (animSelectEl.value !== a) animSelectEl.value = a;
      animPanelEl.classList.toggle("disabled", a !== EDIT);
      readInspector(selectedEid$.value);
    }),
  );
  subs.add(
    fromEvent(animSelectEl, "change").subscribe(() => selectedAnimation$.next(animSelectEl.value)),
  );

  for (const key of poseKeys) {
    subs.add(fromEvent(poseFields[key], "input").subscribe(() => writeInspector()));
  }

  subs.add(clips$.subscribe(() => refreshAnimations()));
  subs.add(
    editClip$.subscribe((edit) => {
      renderRecords(edit);
      if (edit !== null) registerEdit(edit);
    }),
  );

  subs.add(fromEvent(snapshotEl, "click").subscribe(() => snapshot()));
  subs.add(fromEvent(logClipEl, "click").subscribe(() => logClip()));
  subs.add(
    fromEvent(clipNameEl, "input").subscribe(() => {
      const edit = ensureEditClip();
      if (edit === null) return;
      edit.name = clipNameEl.value.trim() || "clip";
      editClip$.next(edit);
    }),
  );
  subs.add(
    fromEvent(clipDurationEl, "input").subscribe(() => {
      const edit = ensureEditClip();
      if (edit === null) return;
      edit.duration = Number(clipDurationEl.value) || edit.duration;
      editClip$.next(edit);
    }),
  );
  subs.add(
    fromEvent<InputEvent>(keysEl, "input").subscribe((e) => {
      const el = e.target as HTMLInputElement;
      if (el.dataset.idx) {
        setRecordField(Number(el.dataset.idx), el.dataset.field as "key" | "pct", Number(el.value));
      }
    }),
  );
  subs.add(
    fromEvent<PointerEvent>(keysEl, "click").subscribe((e) => {
      const el = (e.target as HTMLElement).closest<HTMLElement>(".rm");
      if (el?.dataset.rm) removeRecord(Number(el.dataset.rm));
    }),
  );

  subs.add(
    selectedEntityId$.subscribe((id) => {
      if (selectEl.value !== id) selectEl.value = id;
    }),
  );
  subs.add(fromEvent(selectEl, "change").subscribe(() => selectedEntityId$.next(selectEl.value)));
  subs.add(
    selectedScale$.subscribe((scale) => {
      const v = String(scale);
      if (scaleEl.value !== v) scaleEl.value = v;
    }),
  );
  subs.add(
    fromEvent(scaleEl, "input").subscribe(() => {
      selectedScale$.next(Number(scaleEl.value));
      rebuild$.next();
    }),
  );
  subs.add(fromEvent(regenBtn, "click").subscribe(() => rebuild$.next()));

  subs.add(
    fromEvent<PointerEvent>(treeEl, "click").subscribe((e) => {
      const el = (e.target as HTMLElement).closest<HTMLElement>("[data-eid]");
      if (el) selectedEid$.next(Number(el.dataset.eid));
    }),
  );

  const down$ = fromEvent<PointerEvent>(canvas, "pointerdown");
  const up$ = fromEvent<PointerEvent>(canvas, "pointerup");
  const move$ = fromEvent<PointerEvent>(canvas, "pointermove");

  subs.add(
    down$
      .pipe(
        switchMap((d) => {
          canvas.setPointerCapture(d.pointerId);
          let lastX = d.clientX;
          let lastY = d.clientY;
          return move$.pipe(
            takeUntil(up$),
            map((m) => {
              const dx = m.clientX - lastX;
              const dy = m.clientY - lastY;
              lastX = m.clientX;
              lastY = m.clientY;
              return { dx, dy };
            }),
          );
        }),
      )
      .subscribe(({ dx, dy }) => {
        setCameraAzimuth(cameraAzimuth.value + dx * 0.4);
        setCameraElevation(cameraElevation.value - dy * 0.4);
      }),
  );
  subs.add(up$.subscribe((u) => canvas.releasePointerCapture(u.pointerId)));

  subs.add(
    fromEvent<WheelEvent>(canvas, "wheel", { passive: false }).subscribe((e) => {
      e.preventDefault();
      setCameraZoom(cameraZoom.value * (e.deltaY > 0 ? 0.9 : 1.1));
    }),
  );

  let then = performance.now();
  function loop(now: number): void {
    const delta = Math.min(now - then, 16.6667) / 1000;
    then = now;
    if (!isEdit()) {
      const anim = currentAnimations[selectedAnimation$.value];
      if (anim) {
        anim(delta);
        readInspector(selectedEid$.value);
      }
    }
    engine.tick(delta);
    if (pendingFocus >= 0) {
      focusCamera(pendingFocus);
      pendingFocus = -1;
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="color:#f88;padding:20px">${(err as Error)?.stack ?? err}</pre>`;
  console.error(err);
});
