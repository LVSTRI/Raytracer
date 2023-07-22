#include <filesystem>
#include <iostream>
#include <fstream>
#include <future>
#include <vector>
#include <random>
#include <span>

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <glad/gl.h>

#include <GLFW/glfw3.h>

#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>

using bvh_vec3 = bvh::v2::Vec<float, 3>;
using bvh_aabb = bvh::v2::BBox<float, 3>;
using bvh_prim = bvh::v2::Tri<float, 3>;
using bvh_node = bvh::v2::Node<float, 3>;
using bvh_ray = bvh::v2::Ray<float, 3>;
using bvh_struct = bvh::v2::Bvh<bvh_node>;
using bvh_perm_prim = bvh::v2::PrecomputedTri<float>;

namespace fs = std::filesystem;

struct camera_t {
    glm::vec3 position = { 9.67f, 0.885f, 1.735f };
    float yaw = 201.325f;
    float pitch = 18.75f;
};

struct vertex_t {
    glm::vec3 normal = {};
};

struct material_t {
    float emission_strength = 0.0f;
    glm::vec3 emission_color = {};
    glm::vec3 base_color = {};
};

struct mesh_data_t {
    std::span<const bvh_perm_prim> perm_prims;
    std::span<const vertex_t> attributes;
    std::span<const uint32_t> offsets;
    std::span<material_t> materials;
    std::span<glm::vec4> image;
};

template <typename T>
static auto as_vec3(const T& v) noexcept -> bvh_vec3 {
    return { v.x, v.y, v.z };
}

template <typename F>
static auto intersect(const bvh_struct& bvh, bvh_ray& ray, F&& f) noexcept -> uint32_t {
    constexpr static auto is_any_hit = false;
    constexpr static auto is_robust = false;
    auto hit_id = size_t(-1);
    auto stack = bvh::v2::SmallStack<bvh_struct::Index, 64>();
    auto& root = bvh.get_root();
    bvh.intersect<is_any_hit, is_robust>(ray, root.index, stack, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            if (f(i)) {
                hit_id = i;
            }
        }
        return hit_id != -1;
    });
    return hit_id;
}

static auto __random(uint32_t& state) noexcept -> float {
    state = state * 747796405u + 2891336453u;
    const auto word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return static_cast<float>((word >> 22u) ^ word) / static_cast<float>(0xffffffffu);
}

static auto normal_random(uint32_t& state) noexcept -> float {
    const auto theta = glm::two_pi<float>() * __random(state);
    const auto rho = glm::sqrt(-2.0f * glm::log(__random(state)));
    return rho * glm::cos(theta);
}

static auto random_direction(uint32_t& state) noexcept -> glm::vec3 {
    const auto x = normal_random(state);
    const auto y = normal_random(state);
    const auto z = normal_random(state);
    return glm::normalize(glm::vec3(x, y, z));
}

static auto random_direction_in_hemisphere(uint32_t& state, const glm::vec3& normal) noexcept -> glm::vec3 {
    const auto direction = random_direction(state);
    return direction * glm::sign(glm::dot(normal, direction));
}

static auto whole_file(const fs::path& path) noexcept -> std::string {
    auto file = std::ifstream(path, std::ios::ate);
    auto result = std::string(file.tellg(), '\0');
    file.seekg(0, std::ios::beg);
    file.read(result.data(), result.size());
    return result;
}

static auto environmental_light(const bvh_ray& ray) noexcept -> glm::vec3 {
    static auto k_horizon = glm::vec3(0.757f, 0.761f, 0.824f);
    static auto k_zenith = glm::vec3(0.527f, 0.804f, 0.918f);
    static auto k_ground = glm::vec3(0.276f, 0.268f, 0.281f);
    const auto sky_gradient_t = glm::pow(glm::smoothstep(0.0f, 0.4f, ray.dir[1]), 0.35f);
    const auto ground_to_sky_t = glm::smoothstep(-0.01f, 0.0f, ray.dir[1]);
    const auto sky_gradient = glm::mix(k_horizon, k_zenith, sky_gradient_t);
    return glm::mix(k_ground, sky_gradient, ground_to_sky_t);
}

static auto trace(
    bvh::v2::ParallelExecutor& executor,
    const bvh_struct& bvh,
    const camera_t& camera,
    const mesh_data_t& data,
    uint32_t width,
    uint32_t height,
    uint32_t frame
) noexcept {
    const auto aspect = width / static_cast<float>(height);
    const auto projection = glm::perspective(glm::radians(90.0f), aspect, 0.1f, 100.0f);
    const auto r_yaw = glm::radians(camera.yaw);
    const auto r_pitch = glm::radians(camera.pitch);
    const auto forward = glm::vec3(
        glm::cos(r_yaw) * glm::cos(r_pitch),
        glm::sin(r_pitch),
        glm::sin(r_yaw) * glm::cos(r_pitch));
    const auto view = glm::lookAt(camera.position, camera.position + forward, glm::vec3(0.0f, 1.0f, 0.0f));
    const auto inv_pv = glm::inverse(projection * view);
    const auto spp = 1u;

    executor.for_each(0, height, [&](size_t start, size_t end) {
        for (auto y = start; y < end; ++y) {
        //for (auto y = 0; y < height; ++y) {
            for (auto x = 0u; x < width; ++x) {
                auto state = static_cast<uint32_t>((y * width + x) ^ ((frame + 0xc0c0c0c0u) * 0xfafafafau));
                auto color = glm::vec3(0.0f, 0.0f, 0.0f);
                for (auto s = 0u; s < spp; ++s) {
                    const auto u = glm::clamp((float(x) + __random(state)) / static_cast<float>(width - 1), 0.0f, 1.0f);
                    const auto v = glm::clamp((float(y) + __random(state)) / static_cast<float>(height - 1), 0.0f, 1.0f);
                    const auto uv_near = glm::vec4(glm::vec2(u, v) * 2.0f - 1.0f, 0.0f, 1.0f);
                    const auto uv_far = glm::vec4(glm::vec2(u, v) * 2.0f - 1.0f, 1.0f, 1.0f);
                    auto world_near = inv_pv * uv_near;
                    auto world_far = inv_pv * uv_far;
                    world_near /= world_near.w;
                    world_far /= world_far.w;

                    auto ray = bvh_ray(
                        as_vec3(world_near),
                        as_vec3(glm::normalize(world_far - world_near)),
                        0.001f);
                    auto bary = glm::vec3(0.0f);
                    auto ray_color = glm::vec3(1.0f);
                    auto incoming_light = glm::vec3(0.0f);
                    for (auto bounce = 0u; bounce < 8; ++bounce) {
                        const auto hit = intersect(bvh, ray, [&](size_t i) {
                            if (auto hit = data.perm_prims[bvh.prim_ids[i]].intersect(ray)) {
                                const auto& [b_u, b_v] = *hit;
                                bary = glm::vec3(1.0f - b_u - b_v, b_u, b_v);
                                return true;
                            }
                            return false;
                        });
                        if (hit != -1) {
                            const auto mesh_id = std::distance(
                                data.offsets.begin(),
                                std::ranges::lower_bound(data.offsets, bvh.prim_ids[hit], [](const auto& x, const auto& y) {
                                    return x <= y;
                                }));
                            const auto& n0 = data.attributes[bvh.prim_ids[hit] * 3 + 0].normal;
                            const auto& n1 = data.attributes[bvh.prim_ids[hit] * 3 + 1].normal;
                            const auto& n2 = data.attributes[bvh.prim_ids[hit] * 3 + 2].normal;
                            const auto normal = glm::normalize(
                                n0 * bary.x +
                                n1 * bary.y +
                                n2 * bary.z);
                            const auto point =
                                glm::vec3(ray.org[0], ray.org[1], ray.org[2]) +
                                glm::vec3(ray.dir[0], ray.dir[1], ray.dir[2]) *
                                ray.tmax;
                            auto direction = random_direction_in_hemisphere(state, normal);
                            ray = bvh_ray(as_vec3(point), as_vec3(direction), 0.001f);

                            const auto& material = data.materials[mesh_id];
                            incoming_light += material.emission_color * material.emission_strength * ray_color;
                            ray_color *= material.base_color;
                        } else {
                            //incoming_light += environmental_light(ray);
                            break;
                        }
                    }
                    color += incoming_light;
                }
                color /= spp;
                data.image[y * width + x] = glm::vec4(color, 1.0f);
            }
        }
    });
}

int main() {
    if (!glfwInit()) {
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    static auto width = 1280;
    static auto height = 720;
    auto* window = glfwCreateWindow(width, height, "RT", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGL(glfwGetProcAddress)) {
        glfwTerminate();
        return -1;
    }

#if !defined(NDEBUG)
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback([] (
        GLenum source,
        GLenum type,
        GLuint id,
        GLenum severity,
        GLsizei length,
        const GLchar* message,
        const void*
    ) {
        if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) {
            return;
        }
        std::cout << "debug callback: " << message << std::endl;
        if (severity == GL_DEBUG_SEVERITY_HIGH) {
            std::terminate();
        }
    }, nullptr);
#endif

    static bool is_resized = false;
    glViewport(0, 0, width, height);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* handle, int w, int h) {
        width = w;
        height = h;
        is_resized = true;
        glViewport(0, 0, width, height);
    });
    auto image = std::vector<glm::vec4>(width * height);
    auto camera = camera_t();

    auto vertices = std::vector<vertex_t>();
    auto offsets = std::vector<uint32_t>();
    auto primitives = std::vector<bvh_prim>();
    auto materials = std::vector<material_t>();
    {
        auto options = cgltf_options();
        auto* gltf = static_cast<cgltf_data*>(nullptr);
        const auto* path = "../models/box.glb";
        cgltf_parse_file(&options, path, &gltf);
        cgltf_load_buffers(&options, gltf, path);

        for (auto i = 0u; i < gltf->nodes_count; ++i) {
            const auto& node = gltf->nodes[i];
            if (!node.mesh) {
                continue;
            }
            const auto& mesh = *node.mesh;
            for (auto j = 0u; j < mesh.primitives_count; ++j) {
                const auto& primitive = mesh.primitives[j];
                const auto* position_ptr = (glm::vec3*)(nullptr);
                const auto* normal_ptr = (glm::vec3*)(nullptr);
                auto vertex_count = 0u;
                for (auto k = 0u; k < primitive.attributes_count; ++k) {
                    const auto& attribute = primitive.attributes[k];
                    const auto& accessor = *attribute.data;
                    const auto& buffer_view = *accessor.buffer_view;
                    const auto& buffer = *buffer_view.buffer;
                    const auto& data_ptr = static_cast<const char*>(buffer.data);
                    switch (attribute.type) {
                        case cgltf_attribute_type_position:
                            vertex_count = accessor.count;
                            position_ptr = reinterpret_cast<const glm::vec3*>(data_ptr + buffer_view.offset + accessor.offset);
                            break;

                        case cgltf_attribute_type_normal:
                            normal_ptr = reinterpret_cast<const glm::vec3*>(data_ptr + buffer_view.offset + accessor.offset);
                            break;

                        default: break;
                    }
                }
                if (!position_ptr) {
                    continue;
                }
                auto positions = std::vector<glm::vec3>(vertex_count);
                std::memcpy(positions.data(), position_ptr, vertex_count * sizeof(glm::vec3));
                auto normals = std::vector<glm::vec3>(vertex_count);
                std::memcpy(normals.data(), normal_ptr, vertex_count * sizeof(glm::vec3));

                auto indices = std::vector<uint32_t>();
                {
                    const auto& accessor = *primitive.indices;
                    const auto& buffer_view = *accessor.buffer_view;
                    const auto& buffer = *buffer_view.buffer;
                    const auto& data_ptr = static_cast<const char*>(buffer.data);
                    indices.reserve(accessor.count);
                    switch (accessor.component_type) {
                        case cgltf_component_type_r_8:
                        case cgltf_component_type_r_8u: {
                            const auto* ptr = reinterpret_cast<const uint8_t*>(data_ptr + buffer_view.offset + accessor.offset);
                            std::ranges::copy(std::span(ptr, accessor.count), std::back_inserter(indices));
                        } break;

                        case cgltf_component_type_r_16:
                        case cgltf_component_type_r_16u: {
                            const auto* ptr = reinterpret_cast<const uint16_t*>(data_ptr + buffer_view.offset + accessor.offset);
                            std::ranges::copy(std::span(ptr, accessor.count), std::back_inserter(indices));
                        } break;

                        case cgltf_component_type_r_32f:
                        case cgltf_component_type_r_32u: {
                            const auto* ptr = reinterpret_cast<const uint32_t*>(data_ptr + buffer_view.offset + accessor.offset);
                            std::ranges::copy(std::span(ptr, accessor.count), std::back_inserter(indices));
                        } break;

                        default: break;
                    }
                }
                auto transform = glm::mat4(1.0f);
                cgltf_node_transform_world(&node, &transform[0][0]);
                for (auto w = 0u; w < indices.size() / 3; ++w) {
                    const auto& v0 = transform * glm::vec4(positions[indices[w * 3 + 0]], 1.0f);
                    const auto& v1 = transform * glm::vec4(positions[indices[w * 3 + 1]], 1.0f);
                    const auto& v2 = transform * glm::vec4(positions[indices[w * 3 + 2]], 1.0f);
                    vertices.insert(vertices.end(), {
                        vertex_t { glm::normalize(glm::transpose(glm::inverse(glm::mat3(transform))) * normals[indices[w * 3 + 0]]) },
                        vertex_t { glm::normalize(glm::transpose(glm::inverse(glm::mat3(transform))) * normals[indices[w * 3 + 1]]) },
                        vertex_t { glm::normalize(glm::transpose(glm::inverse(glm::mat3(transform))) * normals[indices[w * 3 + 2]]) },
                    });
                    primitives.emplace_back(
                        bvh_vec3(v0.x, v0.y, v0.z),
                        bvh_vec3(v1.x, v1.y, v1.z),
                        bvh_vec3(v2.x, v2.y, v2.z));
                }

                if (primitive.material) {
                    const auto& material = *primitive.material;
                    auto& data = materials.emplace_back();
                    data.base_color = glm::make_vec4(material.pbr_metallic_roughness.base_color_factor);
                    data.emission_color = glm::make_vec3(material.emissive_factor);
                    data.emission_strength = material.emissive_strength.emissive_strength;
                }
                offsets.emplace_back(primitives.size());
            }
        }
    }

    auto thread_pool = bvh::v2::ThreadPool();
    auto executor = bvh::v2::ParallelExecutor(thread_pool, 512);

    auto prim_bounds = std::vector<bvh_aabb>(primitives.size());
    auto prim_centers = std::vector<bvh_vec3>(primitives.size());
    executor.for_each(0, primitives.size(), [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; i++) {
            prim_bounds[i] = primitives[i].get_bbox();
            prim_centers[i] = primitives[i].get_center();
        }
    });

    auto config = bvh::v2::DefaultBuilder<bvh_node>::Config();
    auto bvh = bvh::v2::DefaultBuilder<bvh_node>::build(thread_pool, prim_bounds, prim_centers, config);

    auto perm_prims = std::vector<bvh_perm_prim>(primitives.size());
    executor.for_each(0, perm_prims.size(), [&] (size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            perm_prims[i] = primitives[i];
        }
    });

    auto textures = std::to_array<uint32_t, 3>({});
    glCreateTextures(GL_TEXTURE_2D, textures.size(), textures.data());
    for (const auto& texture : textures) {
        glTextureStorage2D(texture, 1, GL_RGBA32F, width, height);
        glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    auto framebuffers = std::to_array<uint32_t, 3>({});
    glCreateFramebuffers(framebuffers.size(), framebuffers.data());
    for (auto i = 0u; i < framebuffers.size(); ++i) {
        glNamedFramebufferTexture(framebuffers[i], GL_COLOR_ATTACHMENT0, textures[i], 0);
    }

    auto last_time = 0.0f;
    auto delta_time = 0.0f;
    auto frame_count = 0u;

    auto vao = 0u;
    glCreateVertexArrays(1, &vao);
    auto average_shader = glCreateProgram();
    {
        auto vertex_shader_file = whole_file("../shaders/average.vert");
        auto vertex_shader_source = vertex_shader_file.data();

        auto vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex_shader, 1, &vertex_shader_source, nullptr);
        glCompileShader(vertex_shader);

        auto fragment_shader_file = whole_file("../shaders/average.frag");
        auto fragment_shader_source = fragment_shader_file.data();

        auto fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_shader, 1, &fragment_shader_source, nullptr);
        glCompileShader(fragment_shader);

        glAttachShader(average_shader, vertex_shader);
        glAttachShader(average_shader, fragment_shader);
        glLinkProgram(average_shader);

        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);

        glUseProgram(average_shader);
    }

    auto tonemap_shader = glCreateProgram();
    {
        auto vertex_shader_file = whole_file("../shaders/tonemap.vert");
        auto vertex_shader_source = vertex_shader_file.data();

        auto vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex_shader, 1, &vertex_shader_source, nullptr);
        glCompileShader(vertex_shader);

        auto fragment_shader_file = whole_file("../shaders/tonemap.frag");
        auto fragment_shader_source = fragment_shader_file.data();

        auto fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_shader, 1, &fragment_shader_source, nullptr);
        glCompileShader(fragment_shader);

        glAttachShader(tonemap_shader, vertex_shader);
        glAttachShader(tonemap_shader, fragment_shader);
        glLinkProgram(tonemap_shader);

        assert(glGetError() == GL_NO_ERROR);

        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);

        glUseProgram(tonemap_shader);
    }

    constexpr static auto current_image = 0u;
    constexpr static auto buffer_image = 1u;
    constexpr static auto accumulate_image = 2u;
    glfwSwapInterval(0);
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        auto current_time = glfwGetTime();
        delta_time = current_time - last_time;
        last_time = current_time;

        if (is_resized) {
            glDeleteTextures(textures.size(), textures.data());
            glDeleteFramebuffers(framebuffers.size(), framebuffers.data());

            textures = std::to_array<uint32_t, 3>({});
            glCreateTextures(GL_TEXTURE_2D, textures.size(), textures.data());
            for (const auto& texture : textures) {
                glTextureStorage2D(texture, 1, GL_RGBA32F, width, height);
                glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            }

            framebuffers = std::to_array<uint32_t, 3>({});
            glCreateFramebuffers(framebuffers.size(), framebuffers.data());
            for (auto i = 0u; i < framebuffers.size(); ++i) {
                glNamedFramebufferTexture(framebuffers[i], GL_COLOR_ATTACHMENT0, textures[i], 0);
            }

            image = std::vector<glm::vec4>(width * height);
            is_resized = false;
        }

        constexpr static auto camera_speed = 5.0f;
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
            camera.yaw -= camera_speed * delta_time * 30.0f;
            frame_count = 0;
        }
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
            camera.yaw += camera_speed * delta_time * 30.0f;
            frame_count = 0;
        }
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
            camera.pitch += camera_speed * delta_time * 30.0f;
            frame_count = 0;
        }
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
            camera.pitch -= camera_speed * delta_time * 30.0f;
            frame_count = 0;
        }
        camera.pitch = glm::clamp(camera.pitch, -89.999f, 89.999f);
        const auto r_yaw = glm::radians(camera.yaw);

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            camera.position.x += glm::cos(r_yaw) * camera_speed * delta_time;
            camera.position.z += glm::sin(r_yaw) * camera_speed * delta_time;
            frame_count = 0;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            camera.position.x -= glm::cos(r_yaw) * camera_speed * delta_time;
            camera.position.z -= glm::sin(r_yaw) * camera_speed * delta_time;
            frame_count = 0;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            camera.position.x -= glm::sin(r_yaw) * camera_speed * delta_time;
            camera.position.z += glm::cos(r_yaw) * camera_speed * delta_time;
            frame_count = 0;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            camera.position.x += glm::sin(r_yaw) * camera_speed * delta_time;
            camera.position.z -= glm::cos(r_yaw) * camera_speed * delta_time;
            frame_count = 0;
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            camera.position.y += camera_speed * delta_time;
            frame_count = 0;
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
            camera.position.y -= camera_speed * delta_time;
            frame_count = 0;
        }

        trace(
            executor,
            bvh,
            camera,
            {
            	perm_prims,
            	vertices,
                offsets,
                materials,
            	image,
            },
            width,
            height,
            frame_count);
        glTextureSubImage2D(
            textures[current_image],
            0,
            0,
            0,
            width,
            height,
            GL_RGBA,
            GL_FLOAT,
            image.data());
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffers[accumulate_image]);
        glUseProgram(average_shader);
        glBindTextureUnit(0, textures[buffer_image]);
        glBindTextureUnit(1, textures[current_image]);
        glUniform1i(0, 0);
        glUniform1i(1, 1);
        glUniform1ui(2, frame_count);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glUseProgram(tonemap_shader);
        glBindTextureUnit(0, textures[buffer_image]);
        glUniform1i(0, 0);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glBlitNamedFramebuffer(
            framebuffers[accumulate_image],
            framebuffers[buffer_image],
            0,
            0,
            width,
            height,
            0,
            0,
            width,
            height,
            GL_COLOR_BUFFER_BIT,
            GL_LINEAR);

        glfwSwapBuffers(window);
        frame_count++;
    }

    return 0;
}
