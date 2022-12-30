
pub mod vs_write {
    vulkano_shaders::shader! {
            ty: "vertex",
            src: "
#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec4 color;
layout (location = 2) in vec3 normal;

layout (set = 0, binding = 0) uniform ViewProjection
{
	mat4 view;
	mat4 projection;
} ubo;

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec3 outViewVec;
layout (location = 3) out vec3 outLightVec;

void main()
{
	gl_Position =  ubo.projection * ubo.view * vec4(position, 1.0);
	outColor = color.rgb;
    outNormal = normal;
	outLightVec = vec3(0.0f, 5.0f, 15.0f) - position;
	outViewVec = -position.xyz;
}
"
    }
}

pub mod fs_write {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec3 inViewVec;
layout (location = 3) in vec3 inLightVec;

layout (location = 0) out vec4 outColor;

void main()
{
	// Toon shading color attachment output
	float intensity = dot(normalize(inNormal), normalize(inLightVec));
	float shade = 1.0;
	shade = intensity < 0.5 ? 0.75 : shade;
	shade = intensity < 0.35 ? 0.6 : shade;
	shade = intensity < 0.25 ? 0.5 : shade;
	shade = intensity < 0.1 ? 0.25 : shade;

	outColor.rgb = inColor * 3.0 * shade;

	// Depth attachment does not need to be explicitly written
}
"
    }
}

pub mod vs_read {
    vulkano_shaders::shader! {
            ty: "vertex",
            src: "
#version 450

out gl_PerVertex {
	vec4 gl_Position;
};

void main()
{
    // Creates a triangle with a right angle at (0, 0) that fills the screen
	gl_Position = vec4(vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0f - 1.0f, 0.0f, 1.0f);
}
"
    }
}

pub mod fs_read {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

layout (input_attachment_index = 0, binding = 0) uniform subpassInput inputColor;
layout (input_attachment_index = 1, binding = 1) uniform subpassInput inputDepth;

layout (binding = 2) uniform UBO {
	vec2 brightnessContrast;
	vec2 range;
	int attachmentIndex;
} ubo;

layout (location = 0) out vec4 outColor;

vec3 brightnessContrast(vec3 color, float brightness, float contrast) {
	return (color - 0.5) * contrast + 0.5 + brightness;
}

void main()
{
	// Apply brightness and contrast filer to color input
	if (ubo.attachmentIndex == 0) {
		// Read color from previous color input attachment
		vec3 color = subpassLoad(inputColor).rgb;
		outColor.rgb = brightnessContrast(color, ubo.brightnessContrast[0], ubo.brightnessContrast[1]);
	}

	// Visualize depth input range
	if (ubo.attachmentIndex == 1) {
		// Read depth from previous depth input attachment
		float depth = subpassLoad(inputDepth).r;
		outColor.rgb = vec3((depth - ubo.range[0]) /
		                (ubo.range[1] - ubo.range[0]));
	}
}
"
    }
}
