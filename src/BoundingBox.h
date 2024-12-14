#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

class BoundingBox {
public:
    BoundingBox(float particleDiameter, const glm::vec3& initialDimensions, const glm::vec3& initialPosition);

    void resize(const glm::vec3& newDimensions);
    void rotate(float angleDegrees, const glm::vec3& axis);
    const std::vector<glm::vec3>& getVertices() const;
    glm::mat4 getModelMatrix() const;
    glm::vec3 getDimensions() const;
    glm::vec3 getPosition() const;
    void setPosition(const glm::vec3& newPosition);

private:
    float particleDiameter;
    glm::vec3 dimensions{};
    glm::vec3 position;
    glm::quat rotation{};
    std::vector<glm::vec3> vertices;

    glm::vec3 snapToMultiple(const glm::vec3& vec) const;
    void updateVertices();
};

#endif // BOUNDING_BOX_H
