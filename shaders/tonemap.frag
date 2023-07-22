#version 460 core

layout (location = 0) in vec2 uv;

layout (location = 0) out vec4 o_pixel;

layout (location = 0) uniform sampler2D color;

void main() {
    const vec3 f_color = texture(color, uv).rgb;
    vec3 mapped = f_color / (f_color + vec3(1.0));
    for (uint i = 0; i < 3; ++i) {
        if (mapped[i] <= 0.0031308) {
            mapped[i] = 12.92 * mapped[i];
        } else {
            mapped[i] = 1.055 * pow(mapped[i], 0.41666) - 0.055;
        }
    }
    o_pixel = vec4(mapped, 1.0);
}
