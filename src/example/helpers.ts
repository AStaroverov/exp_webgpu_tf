import { Color, Resolution, Roundness, setColor, Thinness, Translate } from '../ECS/Component/Common.ts';
import { Rope, ROPE_BUFFER_LENGTH } from '../ECS/Component/Rope.ts';
import { canvas } from '../gpu.ts';
import { macroTasks } from '../../lib/TasksScheduler/macroTasks.ts';
import { addComponent, addEntity, IWorld } from 'bitecs';
import { Shape } from '../ECS/Component/Shape.ts';

export function createRopes(world: IWorld) {
    for (let i = 0; i < 100; i++) {
        const id = addEntity(world);
        addComponent(world, Resolution, id);
        addComponent(world, Translate, id);
        addComponent(world, Thinness, id);
        addComponent(world, Color, id);
        addComponent(world, Rope, id);

        const points = Array.from(
            { length: ROPE_BUFFER_LENGTH },
            (_, i) => i % 2 === 0
                /*x*/ ? i * canvas.width / ROPE_BUFFER_LENGTH
                /*y*/ : (0.1 + Math.random() * 0.8) * canvas.height,
        );

        Rope.points[id].set(points);
        Thinness.value[id] = 30;
        setColor(id, Math.random(), Math.random(), Math.random(), 0.5 + Math.random() / 2);

        function update() {
            const points = Array.from(
                { length: ROPE_BUFFER_LENGTH },
                (_, i) => i % 2 === 0
                    /*x*/ ? i * canvas.width / ROPE_BUFFER_LENGTH
                    /*y*/ : (0.1 + Math.random() * 0.8) * canvas.height,
            );

            Rope.points[id].set(points);
        }

        macroTasks.addInterval(update, 3000);
    }
}

export function createShapes(world: IWorld) {
    for (let i = 0; i < 10; i++) {
        const id = addEntity(world);
        addComponent(world, Resolution, id);
        addComponent(world, Translate, id);
        addComponent(world, Roundness, id);
        addComponent(world, Thinness, id);
        addComponent(world, Color, id);
        addComponent(world, Shape, id);

        // Shape.kind[id] = Math.round(3 * Math.random());
        // Shape.point1[id].set([50 + Math.random() * 400, 50 + Math.random() * 400]);
        // Shape.point2[id].set([50 + Math.random() * 400, 50 + Math.random() * 400]);
        // Thinness.value[id] = 10 + 20 * Math.random();
        // Roundness.value[id] = Thinness.value[id] * Math.random();
        // setColor(id, Math.random(), Math.random(), Math.random(), 0.5 + Math.random() / 2);

        // circle
        Translate.x[id] = 50 + Math.random() * 400;
        Translate.y[id] = 50 + Math.random() * 400;
        Shape.kind[id] = 0;
        Thinness.value[id] = 10;
        setColor(id, Math.random(), Math.random(), Math.random(), 0.5 + Math.random() / 2);

        // function update() {
        //     Shape.kind[id] = Math.round(3 * Math.random());
        //     Shape.point1[id].set([50 + Math.random() * 400, 50 + Math.random() * 400]);
        //     Shape.point2[id].set([50 + Math.random() * 400, 50 + Math.random() * 400]);
        //     Thinness.value[id] = 10 + 20 * Math.random();
        //     Roundness.value[id] = Thinness.value[id] * Math.random();
        // }

        // macroTasks.addInterval(update, 1000);
    }
}