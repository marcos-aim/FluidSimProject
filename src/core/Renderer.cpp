#include "Renderer.h"
#include <iostream>

Renderer::Renderer() : boxVAO(0), boxVBO(0), boxEBO(0), sphereVAO(0), sphereVBO(0), sphereEBO(0), instanceVBO(0) {
}

Renderer::~Renderer() {
    if (cudaInstanceResource) {
        cudaGraphicsUnregisterResource(cudaInstanceResource);
    }
    glDeleteVertexArrays(1, &boxVAO);
    glDeleteBuffers(1, &boxVBO);
    glDeleteBuffers(1, &boxEBO);
    glDeleteVertexArrays(1, &sphereVAO);
    glDeleteBuffers(1, &sphereVBO);
    glDeleteBuffers(1, &sphereEBO);
    glDeleteBuffers(1, &instanceVBO);
}


void Renderer::createShaderProgram() {
    // 1) Scene program (box + instanced spheres)
    const char *sceneVertSrc = R"(
    #version 450 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in mat4 instanceModel;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform bool useInstance;
    out vec2 uv;
    void main() {
        mat4 M = useInstance ? instanceModel : model;
        gl_Position = projection * view * M * vec4(aPos,1.0);
        uv = aPos.xy;
    }
    )";

    const char *sceneFragSrc = R"(
    #version 450 core
    in vec2 uv;
    uniform vec4 color;
    uniform vec3 lightDirection;
    uniform bool useInstance;
    out vec4 FragColor;
    void main() {
        if (!useInstance) {
            FragColor = color;
            return;
        }
        float d = length(uv);
        if (d > 1.0) discard;
        vec3 N = normalize(vec3(uv, sqrt(1.0 - d*d)));
        float L = max(dot(N, normalize(lightDirection)),0.0);
        vec3 V = vec3(0,0,1), H = normalize(V + normalize(lightDirection));
        float S = pow(max(dot(N,H),0.0),16.0);
        float A = 0.2;
        vec3 amb = color.rgb * A;
        vec3 col = amb + color.rgb * L + vec3(1.0)*S;
        float alpha = 1.0 - smoothstep(0.95,1.0,d);
        FragColor = vec4(col, color.a * alpha);
    }
    )";

    auto compileAndLink = [&](const char *vsrc, const char *fsrc) {
        GLuint vs = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vs, 1, &vsrc, nullptr);
        glCompileShader(vs);
        GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fs, 1, &fsrc, nullptr);
        glCompileShader(fs);
        GLuint prog = glCreateProgram();
        glAttachShader(prog, vs);
        glAttachShader(prog, fs);
        glLinkProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
        return prog;
    };

    sceneProgram = compileAndLink(sceneVertSrc, sceneFragSrc);
    loadSceneUniformLocations();

    // 2) Texture program (full-screen quad)
    const char *texVertSrc = R"(
    #version 450 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec2 aUV;
    out vec2 uv;
    void main() {
        gl_Position = vec4(aPos.xy,0.0,1.0);
        uv = aUV;
    }
    )";

    const char *texFragSrc = R"(
    #version 450 core
    in vec2 uv;
    uniform sampler2D uCudaTex;
    out vec4 FragColor;
    void main() {
        FragColor = texture(uCudaTex, uv);
    }
    )";

    textureProgram = compileAndLink(texVertSrc, texFragSrc);
    loadTextureUniformLocations();
    initVoxelGridRenderer();
}

void Renderer::loadSceneUniformLocations() {
    modelLoc = glGetUniformLocation(sceneProgram, "model");
    viewLoc = glGetUniformLocation(sceneProgram, "view");
    projectionLoc = glGetUniformLocation(sceneProgram, "projection");
    colorLoc = glGetUniformLocation(sceneProgram, "color");
    lightDirLoc = glGetUniformLocation(sceneProgram, "lightDirection");
    useInstLoc = glGetUniformLocation(sceneProgram, "useInstance");
}

void Renderer::loadTextureUniformLocations() {
    cudaTexLoc = glGetUniformLocation(textureProgram, "uCudaTex");
}

void Renderer::generateBoxData(float width, float height, float depth) {
    boxVertices = {
        0.0f, 0.0f, 0.0f,
        width, 0.0f, 0.0f,
        width, height, 0.0f,
        0.0f, height, 0.0f,
        0.0f, 0.0f, depth,
        width, 0.0f, depth,
        width, height, depth,
        0.0f, height, depth
    };
    boxEdges = {
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
        0, 4, 1, 5, 2, 6, 3, 7
    };
}

void Renderer::prepareBoxBuffers(float width, float height, float depth) {
    generateBoxData(width, height, depth);

    glGenVertexArrays(1, &boxVAO);
    glGenBuffers(1, &boxVBO);
    glGenBuffers(1, &boxEBO);

    glBindVertexArray(boxVAO);

    glBindBuffer(GL_ARRAY_BUFFER, boxVBO);
    glBufferData(GL_ARRAY_BUFFER, boxVertices.size() * sizeof(float), boxVertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boxEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, boxEdges.size() * sizeof(unsigned int), boxEdges.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Renderer::drawBox(const glm::mat4 &view, const glm::mat4 &projection) {
    glUseProgram(sceneProgram);
    glm::mat4 boxModel = glm::mat4(1.0f);
    glUniform1i(useInstLoc, false);
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(boxModel));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniform4f(colorLoc, 1.0f, 1.0f, 1.0f, 1.0f); // White for bounding box

    glBindVertexArray(boxVAO);
    glDrawElements(GL_LINES, boxEdges.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Renderer::generateSphereData(float radius, int slices, int stacks) {
    std::vector<float> vertices;
    for (int i = 0; i <= stacks; ++i) {
        float theta = i * glm::pi<float>() / stacks; // Latitude angle
        float sinTheta = glm::sin(theta);
        float cosTheta = glm::cos(theta);

        for (int j = 0; j <= slices; ++j) {
            float phi = j * 2.0f * glm::pi<float>() / slices; // Longitude angle
            float x = radius * sinTheta * glm::cos(phi);
            float y = radius * cosTheta;
            float z = radius * sinTheta * glm::sin(phi);
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }
    sphereVertices = vertices;

    std::vector<unsigned int> indices;
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            int first = (i * (slices + 1)) + j;
            int second = first + slices + 1;

            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }
    sphereIndices = indices;
}

void Renderer::prepareSphereBuffers(float radius, int slices, int stacks,
                                    const std::vector<glm::mat4> &particleTransforms) {
    generateSphereData(radius, slices, stacks);
    sphereTransforms = particleTransforms;

    glGenVertexArrays(1, &sphereVAO);
    glGenBuffers(1, &sphereVBO);
    glGenBuffers(1, &sphereEBO);

    glBindVertexArray(sphereVAO);

    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, sphereVertices.size() * sizeof(float), sphereVertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereIndices.size() * sizeof(unsigned int), sphereIndices.data(),
                 GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);


    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, particleTransforms.size() * sizeof(glm::mat4), particleTransforms.data(),
                 GL_DYNAMIC_DRAW);

    // Register the instanceVBO with CUDA for direct access.
    cudaError_t cudaStatus = cudaGraphicsGLRegisterBuffer(&cudaInstanceResource, instanceVBO,
                                                          cudaGraphicsRegisterFlagsNone);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to register instanceVBO with CUDA: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    glBindVertexArray(sphereVAO);
    for (int i = 0; i < 4; i++) {
        glEnableVertexAttribArray(1 + i);
        glVertexAttribPointer(1 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *) (i * sizeof(glm::vec4)));
        glVertexAttribDivisor(1 + i, 1);
    }
    glBindVertexArray(0);
}

void Renderer::drawSpheres(const glm::mat4 &view, const glm::mat4 &projection, const glm::vec3 &lightDirection) {
    glUseProgram(sceneProgram);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniform1i(useInstLoc, true);
    glUniform4f(colorLoc, 0.0f, 0.0f, 1.0f, 1.0f);
    // Pass the light direction to the shader
    glUniform3fv(lightDirLoc, 1, glm::value_ptr(lightDirection));

    glBindVertexArray(sphereVAO);
    glDrawElementsInstanced(GL_TRIANGLES, sphereIndices.size(), GL_UNSIGNED_INT, nullptr, sphereTransforms.size());
    glBindVertexArray(0);
}

void Renderer::updateInstanceBuffer(const std::vector<glm::vec3> &updatedPositions) {
    // Create a vector of transformation matrices from particle positions
    std::vector<glm::mat4> updatedTransforms;
    updatedTransforms.reserve(updatedPositions.size());
    for (const auto &pos: updatedPositions) {
        updatedTransforms.push_back(glm::translate(glm::mat4(1.0f), pos));
    }

    // Update the member variable for instance transforms
    sphereTransforms = updatedTransforms;

    // Update the instance VBO with the new transforms
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, updatedTransforms.size() * sizeof(glm::mat4), updatedTransforms.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

#include "RendererKernels.h" // Make sure this include is present at the top if not already

void Renderer::updateInstanceBufferWithCuda(float3 *d_positions, int numParticles) {
    // Map the instance VBO so that CUDA can access it directly.
    cudaError_t err = cudaGraphicsMapResources(1, &cudaInstanceResource, 0);
    if (err != cudaSuccess) {
        std::cerr << "cudaGraphicsMapResources failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    size_t numBytes = 0;
    float *d_instanceTransforms = nullptr;
    err = cudaGraphicsResourceGetMappedPointer((void **) &d_instanceTransforms, &numBytes, cudaInstanceResource);
    if (err != cudaSuccess) {
        std::cerr << "cudaGraphicsResourceGetMappedPointer failed: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnmapResources(1, &cudaInstanceResource, 0);
        return;
    }

    // Launch the kernel to update instance transforms directly into the mapped buffer.
    launchUpdateInstanceTransformsKernel(d_positions, d_instanceTransforms, numParticles);

    // Unmap the resource so OpenGL can use the updated data.
    cudaGraphicsUnmapResources(1, &cudaInstanceResource, 0);
}

void Renderer::initCudaInterop(int width, int height) {
    texWidth = width;
    texHeight = height;

    // 1) Create a GL texture
    glGenTextures(1, &cudaTexture);
    glBindTexture(GL_TEXTURE_2D, cudaTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                 width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    // 2) Register it with CUDA
    cudaGraphicsGLRegisterImage(
        &cudaTextureResource,
        cudaTexture,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsSurfaceLoadStore
    );
}

cudaSurfaceObject_t Renderer::mapCudaSurface() {
    cudaGraphicsMapResources(1, &cudaTextureResource, 0);
    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(
        &array, cudaTextureResource, 0, 0
    );
    cudaResourceDesc desc = {};
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = array;
    cudaSurfaceObject_t surf = 0;
    cudaCreateSurfaceObject(&surf, &desc);
    return surf;
}

void Renderer::unmapCudaSurface(cudaSurfaceObject_t surf) {
    cudaDestroySurfaceObject(surf);
    cudaGraphicsUnmapResources(1, &cudaTextureResource, 0);
}

void Renderer::prepareScreenQuad() {
    float quadVerts[] = {
        -1, +1, 0, 1,
        -1, -1, 0, 0,
        +1, -1, 1, 0,
        +1, +1, 1, 1,
    };
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *) 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *) (2 * sizeof(float)));
    glBindVertexArray(0);
}

void Renderer::drawScreenQuad() {
    // set shader to “texture mode”
    glUseProgram(textureProgram);

    // bind the CUDA-produced texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, cudaTexture);
    glUniform1i(cudaTexLoc, 0);

    // draw the full-screen quad
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::initVoxelGridRenderer() {
    // 1) Shader sources
    const char *vertexSource = R"glsl(
        #version 450 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec4 aInstance;
        layout(location = 2) in float aOpacity;
        layout(location = 3) in vec3 aColor;
        uniform mat4 view;
        uniform mat4 projection;
        out float vOpacity;
        out vec3  vColor;
        void main(){
            vec3 world = aPos * aInstance.w + aInstance.xyz;
            gl_Position = projection * view * vec4(world, 1.0);
            vOpacity = aOpacity;
            vColor   = aColor;
        }
    )glsl";

    const char *fragmentSource = R"glsl(
        #version 450 core
        in float vOpacity;
        in vec3  vColor;
        out vec4 FragColor;
        void main(){
            if (vOpacity <= 0.0) discard;
            FragColor = vec4(vColor, vOpacity);
        }
    )glsl";

    // 2) Compile shaders
    auto compileShader = [&](GLenum type, const char *src) -> GLuint {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        // (Optional: check compile status here)
        return s;
    };

    GLuint vertShader = compileShader(GL_VERTEX_SHADER, vertexSource);
    GLuint fragShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);

    // 3) Link program
    voxelProgram = glCreateProgram();
    glAttachShader(voxelProgram, vertShader);
    glAttachShader(voxelProgram, fragShader);
    glLinkProgram(voxelProgram);
    // (Optional: check link status here)

    // 4) Cleanup shaders
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    // 5) Get uniform locations
    voxelViewLoc = glGetUniformLocation(voxelProgram, "view");
    voxelProjLoc = glGetUniformLocation(voxelProgram, "projection");
    // we don’t have a model‐matrix uniform, so no voxelModelLoc

    // 6) Build a unit cube VAO/VBO/EBO
    float cubeVerts[] = {
        -0.5f, -0.5f, -0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, -0.5f, 0.5f, -0.5f,
        -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f
    };
    unsigned int cubeIdx[] = {
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        0, 4, 7, 7, 3, 0,
        1, 5, 6, 6, 2, 1,
        3, 2, 6, 6, 7, 3,
        0, 1, 5, 5, 4, 0
    };

    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO);
    glGenBuffers(1, &cubeEBO);
    glBindVertexArray(cubeVAO);

    // Positions
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVerts), cubeVerts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) 0);
    glEnableVertexAttribArray(0);

    // Indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cubeIdx), cubeIdx, GL_STATIC_DRAW);

    // 7) Instancing buffers
    glGenBuffers(1, &instVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *) 0);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);

    glGenBuffers(1, &opacVBO);
    glBindBuffer(GL_ARRAY_BUFFER, opacVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void *) 0);
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);

    // 8) color buffer (per-instance RGB)
    glGenBuffers(1, &colorVBO);
    glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) 0);
    glEnableVertexAttribArray(3);
    glVertexAttribDivisor(3, 1);

    glBindVertexArray(0);
}

void Renderer::renderVoxelGrid(
    const UserInput &ui,
    SPHSimulation &sim,
    const glm::mat4 &view,
    const glm::mat4 &projection) {
    sim.updateDensityGrid();
    auto gx = sim.gridDims.x, gy = sim.gridDims.y, gz = sim.gridDims.z;
    size_t total = size_t(gx) * gy * gz;
    if (!gx || !gy || !gz || total > 200000000) {
        drawBox(view, projection);
        return;
    }

    static std::vector<float> dens;
    dens.resize(total);
    sim.downloadDensityGrid(dens);
    if (dens.size() != total) {
        drawBox(view, projection);
        return;
    }

    int skip = std::max(1, int(std::ceil(std::cbrt(float(total) / 200000.0f))));
    static std::vector<glm::vec4> inst;
    inst.clear();
    static std::vector<float> opac;
    opac.clear();
    static std::vector<glm::vec3> colr;
    colr.clear();
    inst.reserve(total / (skip * skip * skip));
    opac.reserve(inst.capacity());
    colr.reserve(inst.capacity());

    const float cs = sim.cellSize,
            refD = ui.restingDensity > 0 ? ui.restingDensity : 1.0f,
            minO = 0.05f, maxO = 0.8f, gamma = 2.0f, fade = 0.5f;
    const glm::vec3 green{0, 1, 0}, lightG{0.5f, 1, 0.5f}, yellow{1, 1, 0},
            red{1, 0, 0}, white{1, 1, 1};

    for (size_t iz = 0; iz < gz; iz += skip) {
        size_t baseZ = iz * gy * gx;
        float zc = (iz + 0.5f) * cs;
        for (size_t iy = 0; iy < gy; iy += skip) {
            size_t baseYZ = baseZ + iy * gx;
            float yc = (iy + 0.5f) * cs;
            for (size_t ix = 0; ix < gx; ix += skip) {
                float d = dens[baseYZ + ix];
                float norm = std::clamp(d / refD, 0.0f, 1.0f);
                float o = (minO + (maxO - minO) * std::pow(norm, gamma)) * fade;
                if (o <= minO) continue;
                inst.emplace_back((ix + 0.5f) * cs, yc, zc, cs);
                opac.push_back(o);

                if (ui.checkResting) {
                    float diff = d - ui.restingDensity,
                            r = std::clamp(std::abs(diff) / ui.restingDensity, 0.0f, 1.0f);
                    glm::vec3 c = diff >= 0
                                      ? (r < 0.5f
                                             ? glm::mix(green, yellow, r * 2.0f)
                                             : glm::mix(yellow, red, (r - 0.5f) * 2.0f))
                                      : (r < 0.5f
                                             ? glm::mix(green, lightG, r * 2.0f)
                                             : glm::mix(lightG, white, (r - 0.5f) * 2.0f));
                    colr.push_back(c);
                } else colr.emplace_back(0.8f);
            }
        }
    }
    if (inst.empty()) {
        drawBox(view, projection);
        return;
    }

    auto upload = [&](GLuint vbo, const void *data, size_t b) {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, b, nullptr,GL_DYNAMIC_DRAW);
        void *p = glMapBufferRange(GL_ARRAY_BUFFER, 0, b,GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        std::memcpy(p, data, b);
        glUnmapBuffer(GL_ARRAY_BUFFER);
    };
    upload(instVBO, inst.data(), inst.size() * sizeof(glm::vec4));
    upload(opacVBO, opac.data(), inst.size() * sizeof(float));
    upload(colorVBO, colr.data(), inst.size() * sizeof(glm::vec3));

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    glUseProgram(voxelProgram);
    glUniformMatrix4fv(voxelViewLoc, 1,GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(voxelProjLoc, 1,GL_FALSE, glm::value_ptr(projection));
    glBindVertexArray(cubeVAO);
    glDrawElementsInstanced(GL_TRIANGLES, 36,GL_UNSIGNED_INT, nullptr, GLsizei(inst.size()));
    glBindVertexArray(0);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    drawBox(view, projection);
}
