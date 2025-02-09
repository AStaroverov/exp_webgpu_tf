import { mat2d, vec2 } from 'gl-matrix';
import { RigidBody } from '@dimforge/rapier2d';

const pivot = vec2.create();
const rotationMatrix = mat2d.create();

export function rotateRigidBodyAroundPoint(body: RigidBody, x: number, y: number, angle: number): void {
    pivot[0] = x;
    pivot[1] = y;
    mat2d.identity(rotationMatrix);
    mat2d.fromRotation(rotationMatrix, angle); // 2D rotation matrix

    const position = body.translation(); // Get current position
    const posVec = vec2.fromValues(position.x, position.y);

    // Translate position relative to pivot
    vec2.subtract(posVec, posVec, pivot);

    // Rotate the position vector
    vec2.transformMat2d(posVec, posVec, rotationMatrix);

    // Translate back
    vec2.add(posVec, posVec, pivot);

    // Apply new position to the rigid body
    body.setTranslation({ x: posVec[0], y: posVec[1] }, true);

    // Update rotation angle
    const currentAngle = body.rotation();
    body.setRotation(currentAngle + angle, true);
}
