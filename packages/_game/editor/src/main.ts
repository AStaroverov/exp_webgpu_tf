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
  cameraZoom,
  setCameraAzimuth,
  setCameraElevation,
  setCameraPosition,
  setCameraZoom,
} from "../../../renderer/src/ECS/Systems/ResizeSystem.ts";
import { ENTITIES, type EntityAnimations } from "./Entities/registry.ts";
import { selectedAnimation$, selectedEid$, selectedEntityId$, selectedScale$ } from "./state.ts";

const NONE = "none";

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const selectEl = document.getElementById("entity-select") as HTMLSelectElement;
  const animSelectEl = document.getElementById("animation-select") as HTMLSelectElement;
  const scaleEl = document.getElementById("scale-input") as HTMLInputElement;
  const regenBtn = document.getElementById("regen") as HTMLButtonElement;
  const treeEl = document.getElementById("tree") as HTMLElement;
  const componentsEl = document.getElementById("components") as HTMLElement;

  const engine = await createEngine({ canvas });
  const world = engine.world as EngineWorld;
  const sceneRoot = engine.sceneRoot;

  const components = getEngineComponents(world);
  const componentEntries = Object.entries(components) as Array<
    [string, EngineComponents[keyof EngineComponents]]
  >;

  const subs = new Subscription();
  const rows: HTMLElement[] = [];

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
        const child = Children.entitiesIds.get(eid, i);
        appendNode(child, nodeLabel(child), depth + 1);
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

  SunLight.enabled = true;
  SunLight.angle = 2.4;
  SunLight.elevation = 0.95;
  SunLight.intensity = 0.9;
  SunLight.color = [1.0, 0.93, 0.82];

  setCameraPosition(0, 2.5);
  cameraZoom.value = 48;
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
  let currentAnimations: EntityAnimations = {};
  function build(id: string): void {
    if (currentRoot >= 0) {
      components.Children.removeChild(sceneRoot, currentRoot);
      removeEntityTree(world, currentRoot);
    }
    const def = ENTITIES.find((d) => d.id === id) ?? ENTITIES[0];
    const instance = def.build(world, { scale: selectedScale$.value });
    currentRoot = instance.root;
    components.Children.addChild(sceneRoot, currentRoot);
    currentAnimations = instance.animations;
    fillAnimationOptions();
    renderTree(currentRoot, def.label);
    selectedEid$.next(currentRoot);
  }

  function fillAnimationOptions(): void {
    const names = [NONE, ...Object.keys(currentAnimations)];
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
    }),
  );

  subs.add(
    selectedEntityId$.subscribe((id) => {
      if (selectEl.value !== id) selectEl.value = id;
    }),
  );
  subs.add(fromEvent(selectEl, "change").subscribe(() => selectedEntityId$.next(selectEl.value)));
  subs.add(
    fromEvent(animSelectEl, "change").subscribe(() => selectedAnimation$.next(animSelectEl.value)),
  );
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
    const anim = currentAnimations[selectedAnimation$.value];
    if (anim) anim(delta);
    engine.tick(delta);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="color:#f88;padding:20px">${(err as Error)?.stack ?? err}</pre>`;
  console.error(err);
});
