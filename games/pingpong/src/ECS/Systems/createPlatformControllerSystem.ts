import { PhysicalRef } from '../Components/Physical.ts';
import { PhysicalWorld } from '../../index.ts';

export function createPlatformControllerSystem(physicalWorld: PhysicalWorld, canvas: HTMLCanvasElement, platformId: number) {
    const maxY = canvas.height - 100;
    const minY = canvas.height - 50;

    let x: number = 0;
    let isJumping = false;

    canvas.addEventListener('mousemove', (e) => {
        x = Math.max(0, Math.min(canvas.width, e.clientX));
    });
    canvas.addEventListener('click', () => {
        isJumping = true;
    });

    return function syncRigidBodyToRenderTransformSystem() {
        const physicalId = PhysicalRef.id[platformId];
        const rigidBody = physicalWorld.getRigidBody(physicalId);
        const position = rigidBody.translation();
        position.x = x;

        if (isJumping) {
            position.y = Math.max(maxY, position.y - 3);

            if (position.y <= maxY) {
                isJumping = false;
            }
        } else if (position.y <= minY) {
            position.y += 3;
        }

        rigidBody.setNextKinematicTranslation(position);
    };
}