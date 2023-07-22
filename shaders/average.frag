#version 460 core

layout (location = 0) in vec2 uv;

layout (location = 0) out vec4 o_pixel;

layout (location = 0) uniform sampler2D color[2];
layout (location = 2) uniform uint frame;

void main() {
    const vec3 old = texture(color[0], uv).rgb;
    const vec3 new = texture(color[1], uv).rgb;

    const float weight = frame <= 100 ?
        1.0 / float(frame + 1) :
        0.01;
    vec3 color = frame == 0 ? new : old;
    for (uint i = 0; i < 3; ++i) {
        if (new[i] > 0.0001) {
            color[i] = new[i];
        }
    }
    //o_pixel = vec4(mix(old, color, weight), 1.0);
    o_pixel = vec4(mix(old, new, weight), 1.0);
}
