import { SystemGroup } from '../ECS/Plugins/systems.js';

export const PluginDI = {
    systems: {
        [SystemGroup.After]: [] as ((delta: number) => void)[],
        [SystemGroup.Before]: [] as ((delta: number) => void)[],
    },
    disposes: [] as VoidFunction[],
    addSystem(group: SystemGroup, system: (delta: number) => void, dispose?: VoidFunction) {
        PluginDI.systems[group].push(system);
        dispose && PluginDI.disposes.push(dispose);
    },
    addDestroy(destroy: VoidFunction) {
        PluginDI.disposes.push(destroy);
    },
    dispose() {
        PluginDI.disposes.forEach(dispose => dispose());
        PluginDI.disposes = [];
        PluginDI.systems = {
            [SystemGroup.After]: [],
            [SystemGroup.Before]: [],
        };
    },
};