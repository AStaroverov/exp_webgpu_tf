import { RigidBody } from '@dimforge/rapier2d';
import { mat4, vec2 } from 'gl-matrix';

export function applyMatrixToRigidBody(body: RigidBody, matrix: mat4): void {
    // Get current local translation (relative to parent)
    const localPosition = body.translation();
    const localVec = vec2.fromValues(localPosition.x, localPosition.y);

    // Convert local position to homogeneous coordinates (x, y, 0, 1)
    const localPosition4D = vec2.transformMat4(vec2.create(), localVec, matrix);

    // Extract rotation from the matrix
    const localAngle = body.rotation();
    const parentRotation = Math.atan2(matrix[4], matrix[0]); // Extract rotation from mat4
    const newAngle = localAngle + parentRotation; // Combine local and parent rotation

    // Apply modified transformation to the body
    body.setTranslation({ x: localPosition4D[0], y: localPosition4D[1] }, true);
    body.setRotation(newAngle, true);
}
