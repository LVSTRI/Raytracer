#version 460 core

layout (location = 0) in vec2 uv;

layout (location = 0) out vec4 o_pixel;

layout (location = 0) uniform sampler2D color[2];
layout (location = 2) uniform uint frame;

void main() {
    const vec3 old = texture(color[0], uv).rgb;
    const vec3 new = texture(color[1], uv).rgb;

    const float weight = 1 / float(frame + 1);
    o_pixel = vec4(mix(old, new, max(weight, 0.01)), 1.0);
}
