import GUI from 'lil-gui';
import { RenderDI } from '../Game/DI/RenderDI.ts';
import { DEFAULT_RC_PARAMS } from '../Game/ECS/Systems/Render/Lighting/createRadianceCascadesSystem.ts';

/**
 * Standalone "Lighting" tuning panel. Holds its own params object so it survives
 * lighting-system recreation (ppo_unknown creates a new scenario per episode):
 * call `sync()` periodically — it re-applies the panel values to the current
 * RenderDI.lighting whenever the instance changes.
 */
export function createLightingGUI({ container, side = 'right' }: {
    container?: HTMLElement,
    side?: 'left' | 'right',
} = {}) {
    const params = structuredClone(DEFAULT_RC_PARAMS);
    const apply = () => RenderDI.lighting?.setParams(params);

    const gui = new GUI({ title: 'Lighting', autoPlace: !container });
    if (container) {
        Object.assign(gui.domElement.style, {
            position: 'fixed', [side]: '0', top: '0', maxHeight: '100vh', overflowY: 'auto', zIndex: '1000',
        });
        container.appendChild(gui.domElement);
    }

    // The directional source: moon, sun, burning city — whatever the scene calls for.
    const source = gui.addFolder('Light source');
    source.add(params, 'enableSun').name('enabled').onChange(apply);
    source.add(params, 'sunAngle', 0, Math.PI * 2, 0.01).name('angle').onChange(apply);
    source.addColor(params, 'sunColor').name('source color').onChange(apply);
    source.add(params, 'sunIntensity', 0, 5, 0.05).name('source intensity').onChange(apply);
    source.addColor(params, 'skyColor').name('sky color').onChange(apply);
    source.add(params, 'skyMix', 0, 1, 0.01).name('sky mix').onChange(apply);
    source.add(params, 'sunDistance', 0.1, 4, 0.05).name('softness (sunDistance)').onChange(apply);

    const mix = gui.addFolder('Mix');
    mix.add(params, 'ambient', 0, 1, 0.01).name('ambient').onChange(apply);
    mix.add(params, 'objectLightRadius', 0, 12, 0.5).name('object light radius').onChange(apply);
    mix.add(params, 'objectTranslucency', 0, 1, 0.01).name('object translucency').onChange(apply);
    mix.add(params, 'srgb', 1, 2.6, 0.05).onChange(apply);
    mix.add(params, 'emitCone', 0, 64, 1).name('emit cone').onChange(apply);

    const rays = gui.addFolder('Rays');
    rays.add(params, 'rayInterval', 0.25, 4, 0.05).onChange(apply);
    rays.add(params, 'intervalOverlap', 0, 0.5, 0.01).onChange(apply);
    rays.close();

    let lastLighting = RenderDI.lighting;
    apply();

    // Re-apply panel values when the lighting system is recreated (new episode/render target).
    function sync() {
        if (RenderDI.lighting !== lastLighting) {
            lastLighting = RenderDI.lighting;
            apply();
        }
    }

    return { gui, sync };
}
