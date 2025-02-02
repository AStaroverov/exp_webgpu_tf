import {uniq} from "../utils.ts";

import {TWGSLModule} from "./def.ts";

export function wgsl(
    strings: TemplateStringsArray,
    ...values: (number | string | TWGSLModule)[]
): TWGSLModule {
    const deps: TWGSLModule[] = [];
    const body = strings.reduce((acc, str, i) => {
        const value = values[i];
        let valueStr = '';

        if (value === undefined || typeof value === 'string' || typeof value === 'number') {
            valueStr = value ? String(value) : '';
        }

        if (typeof value === 'object') {
            if ('name' in value) {
                valueStr = 'name' in value ? value.name : '';
                deps.push(value);
            } else {
                valueStr = value.body ?? '';
            }

            deps.push(...value.deps);
        }

        return acc + str + valueStr;
    }, '');

    return {
        deps: uniq(deps),
        body,
    };
}

const getNextName = () => {
    return 'fn_' + crypto.randomUUID();
};

export function funcWGSL<Dep extends TWGSLModule, Deps extends Dep[]>(
    fn: (...deps: Deps) => (name: string) => TWGSLModule,
    name: string = getNextName(),
): (...deps: Deps) => TWGSLModule {
    return (...deps: Deps): TWGSLModule => {
        return {
            ...fn(...deps)(name),
            name,
        };
    };
}
