import { Color, Resolution, Roundness, setColor, Size, Translate } from '../ECS/Component/Common.ts';
import { Rope, ROPE_BUFFER_LENGTH } from '../ECS/Component/Rope.ts';
import { canvas } from '../gpu.ts';
import { macroTasks } from '../../lib/TasksScheduler/macroTasks.ts';
import { addComponent, addEntity, IWorld } from 'bitecs';
import { setCircle, setParallelogram, setRectangle, setTrapezoid, setTriangle, Shape } from '../ECS/Component/Shape.ts';
import { Transform } from '../ECS/Component/Transform.ts';
import { mat4 } from 'gl-matrix';
import { MAX_INSTANCE_COUNT } from '../ECS/System/SDFSystem/sdf.shader.ts';

export function createRopes(world: IWorld) {
    for (let i = 0; i < 100; i++) {
        const id = addEntity(world);
        addComponent(world, Resolution, id);
        addComponent(world, Translate, id);
        addComponent(world, Color, id);
        addComponent(world, Size, id);
        addComponent(world, Rope, id);

        const points = Array.from(
            { length: ROPE_BUFFER_LENGTH },
            (_, i) => i % 2 === 0
                /*x*/ ? i * canvas.width / ROPE_BUFFER_LENGTH
                /*y*/ : (0.1 + Math.random() * 0.8) * canvas.height,
        );

        Rope.points[id].set(points);
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
    // for (let i = 0; i < 100; i++) {
    for (let i = 0; i < MAX_INSTANCE_COUNT; i++) {
        const id = addEntity(world);
        addComponent(world, Resolution, id);
        addComponent(world, Translate, id);
        addComponent(world, Transform, id);
        addComponent(world, Roundness, id);
        addComponent(world, Color, id);
        addComponent(world, Size, id);
        addComponent(world, Shape, id);

        mat4.identity(Transform.matrix[id]);
        mat4.translate(Transform.matrix[id], Transform.matrix[id], [50 + Math.random() * 400, 50 + Math.random() * 400, 0]);
        // mat4.translate(Transform.matrix[id], Transform.matrix[id], [300, 300, 0]);

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
