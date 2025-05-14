import { RigidBody } from '@dimforge/rapier2d-simd';
import { mat4 } from 'gl-matrix';

const position = { x: 0, y: 0 };

export function setMatrixToRigidBody(body: RigidBody, matrix: mat4): void {
    // Apply transformation to the rigid body
    position.x = matrix[12];
    position.y = matrix[13];

    if (body.isKinematic()) {
        body.setNextKinematicTranslation(position);
        body.setNextKinematicRotation(Math.atan2(matrix[4], matrix[0]));
    } else {
        body.setTranslation(position, true);
        body.setRotation(Math.atan2(matrix[4], matrix[0]), true);
    }
    // body.setTranslation(position, true);
    // body.setRotation(Math.atan2(matrix[4], matrix[0]), true);
}