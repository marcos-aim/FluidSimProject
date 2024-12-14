#include "BoundingBox.h"
#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <cmath>

BoundingBox::BoundingBox(float particleDiameter, const glm::vec3& initialDimensions, const glm::vec3& initialPosition)
    : particleDiameter(particleDiameter), position(initialPosition)
{
    dimensions = snapToMultiple(initialDimensions);
    rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // Identity quaternion
    updateVertices();
}

void BoundingBox::resize(const glm::vec3& newDimensions) {
    dimensions = snapToMultiple(newDimensions);
    updateVertices();
}

void BoundingBox::rotate(float angleDegrees, const glm::vec3& axis) {
    glm::vec3 normalizedAxis = glm::normalize(axis);
    glm::quat deltaRotation = glm::angleAxis(glm::radians(angleDegrees), normalizedAxis);
    rotation = glm::normalize(deltaRotation * rotation);
    updateVertices();
}

const std::vector<glm::vec3>& BoundingBox::getVertices() const {
    return vertices;
}

glm::mat4 BoundingBox::getModelMatrix() const {
    glm::mat4 model = translate(glm::mat4(1.0f), position);
    model *= mat4_cast(rotation); // Apply rotation
    model = scale(model, dimensions);
    return model;
}

glm::vec3 BoundingBox::getDimensions() const {
    return dimensions;
}

glm::vec3 BoundingBox::getPosition() const {
    return position;
}

void BoundingBox::setPosition(const glm::vec3& newPosition) {
    position = newPosition;
    updateVertices();
}

glm::vec3 BoundingBox::snapToMultiple(const glm::vec3& vec) const {
    return {
        std::round(vec.x / particleDiameter) * particleDiameter,
        std::round(vec.y / particleDiameter) * particleDiameter,
        std::round(vec.z / particleDiameter) * particleDiameter
    };
}

void BoundingBox::updateVertices() {
    // Define corners of a unit cube centered at the origin
    std::vector<glm::vec3> unitVertices = {
        {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f},
        {-0.5f,  0.5f, -0.5f}, {0.5f,  0.5f, -0.5f},
        {-0.5f, -0.5f,  0.5f}, {0.5f, -0.5f,  0.5f},
        {-0.5f,  0.5f,  0.5f}, {0.5f,  0.5f,  0.5f}
    };

    glm::mat4 transform = getModelMatrix();
    vertices.clear();
    for (const auto& vertex : unitVertices) {
        vertices.push_back(glm::vec3(transform * glm::vec4(vertex, 1.0f)));
    }
}