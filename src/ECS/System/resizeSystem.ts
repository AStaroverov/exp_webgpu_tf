import {defineQuery, IWorld} from "bitecs";
import {Resolution} from "../Component/Common.ts";

export function createResizeSystem(world: IWorld, canvas: HTMLCanvasElement) {
    const query = defineQuery([Resolution])

    return function resizeSystem() {
        const entities = query(world)

        for (let i = 0; i < entities.length; i++) {
            const id = entities[i];

            Resolution.x[id] = canvas.width
            Resolution.y[id] = canvas.height
        }
    }
}