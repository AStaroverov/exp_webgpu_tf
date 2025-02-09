import { Color, Roundness, setColor, Thinness } from '../src/ECS/Components/Common.ts';
import { Rope, ROPE_BUFFER_LENGTH, ROPE_POINTS_COUNT } from '../src/ECS/Components/Rope.ts';
import { macroTasks } from '../lib/TasksScheduler/macroTasks.ts';
import { addComponent, addEntity, IWorld } from 'bitecs';
import {
    setCircle,
    setParallelogram,
    setRectangle,
    setTrapezoid,
    setTriangle,
    Shape,
} from '../src/ECS/Components/Shape.ts';
import { LocalTransform } from '../src/ECS/Components/Transform.ts';
import { mat4 } from 'gl-matrix';

export function createRopes(world: IWorld, canvas: HTMLCanvasElement) {
    for (let i = 0; i < 200; i++) {
        const id = addEntity(world);
        addComponent(world, LocalTransform, id);
        addComponent(world, Thinness, id);
        addComponent(world, Color, id);
        addComponent(world, Rope, id);

        mat4.identity(LocalTransform.local[id]);
        mat4.translate(
            LocalTransform.local[id],
            LocalTransform.local[id],
            // [0, 0, 0],
            [-50 + Math.random() * 200, -50 + Math.random() * 200, 0],
        );

        const pointsLength = Math.round(ROPE_POINTS_COUNT * Math.random());
        const burfferLenght = pointsLength * 2;

        const points = Array.from(
            { length: burfferLenght },
            (_, i) => i % 2 === 0
                /*x*/ ? i * canvas.width / ROPE_BUFFER_LENGTH
                /*y*/ : (0.1 + Math.random() * 0.8) * canvas.height,
        );

        Rope.points[id].set(points);
        Thinness.value[id] = 1 + Math.random() * 10;
        setColor(id, Math.random(), Math.random(), Math.random(), 0.5 + Math.random() / 2);

        function update() {
            const points = Array.from(
                { length: burfferLenght },
                (_, i) => i % 2 === 0
                    /*x*/ ? i * canvas.width / ROPE_BUFFER_LENGTH
                    /*y*/ : (0.1 + Math.random() * 0.8) * canvas.height,
            );

            Rope.points[id].set(points);
        }

        macroTasks.addInterval(update, 100);
    }
}

export function createShapes(world: IWorld) {
    for (let i = 0; i < 200; i++) {
        const id = addEntity(world);
        addComponent(world, LocalTransform, id);
        addComponent(world, Roundness, id);
        addComponent(world, Color, id);
        addComponent(world, Shape, id);

        mat4.identity(LocalTransform.local[id]);
        mat4.translate(LocalTransform.local[id], LocalTransform.local[id], [50 + Math.random() * 400, 50 + Math.random() * 400, 0]);

        switch (Math.round(4 * Math.random())) {
            case 0:
                setCircle(id, 10 + Math.random() * 50);
                break;
            case 1:
                setRectangle(id, 50 + Math.random() * 100, 50 + Math.random() * 100);
                break;
            case 2:
                setParallelogram(
                    id,
                    50 + Math.random() * 100,
                    50 + Math.random() * 100,
                    -50 + Math.random() * 100,
                );
                break;
            case 3:
                setTrapezoid(
                    id,
                    50 + Math.random() * 100,
                    50 + Math.random() * 100,
                    50 + Math.random() * 100,
                );
                break;
            case 4:
                setTriangle(
                    id,
                    50 + Math.random() * 100,
                    50 + Math.random() * 100,
                    -50 + Math.random() * 100,
                    -50 + Math.random() * 100,
                    50 + Math.random() * 100,
                    -50 + Math.random() * 100,
                );
                break;
        }

        setColor(id, Math.random(), Math.random(), Math.random(), 0.5 + Math.random() / 2);
        Roundness.value[id] = 2;

        function update() {
            switch (Math.round(4 * Math.random())) {
                case 0:
                    setCircle(id, 10 + Math.random() * 50);
                    break;
                case 1:
                    setRectangle(id, 50 + Math.random() * 100, 50 + Math.random() * 100);
                    break;
                case 2:
                    setParallelogram(
                        id,
                        50 + Math.random() * 100,
                        50 + Math.random() * 100,
                        -50 + Math.random() * 100,
                    );
                    break;
                case 3:
                    setTrapezoid(
                        id,
                        50 + Math.random() * 100,
                        50 + Math.random() * 100,
                        50 + Math.random() * 100,
                    );
                    break;
                case 4:
                    setTriangle(
                        id,
                        50 + Math.random() * 100,
                        50 + Math.random() * 100,
                        -50 + Math.random() * 100,
                        -50 + Math.random() * 100,
                        50 + Math.random() * 100,
                        -50 + Math.random() * 100,
                    );
                    break;
            }
        }

        macroTasks.addInterval(update, 100);
    }
}
