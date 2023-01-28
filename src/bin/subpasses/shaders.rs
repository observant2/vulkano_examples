pub mod vs_gbuffer {
    vulkano_shaders::shader! {
            ty: "vertex",
            src: "
#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec4 color;
layout (location = 2) in vec3 normal;

layout (set = 0, binding = 0) uniform ViewProjection
{
    mat4 model;
	mat4 view;
	mat4 projection;
} ubo;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outColor;
layout (location = 2) out vec3 outWorldPos;

out gl_PerVertex
{
	vec4 gl_Position;
};

void main()
{
	gl_Position = ubo.projection * ubo.view * ubo.model * vec4(position, 1.0);

	// Vertex position in world space
	outWorldPos = vec3(ubo.model * vec4(position, 1.0));
	// GL to Vulkan coord space
	outWorldPos.y = -outWorldPos.y;

	// Normal in world space
	mat3 mNormal = transpose(inverse(mat3(ubo.model)));
	outNormal = mNormal * normalize(normal);

	// Currently just vertex color
	outColor = vec3(color);
}
"
    }
}

pub mod fs_gbuffer {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec3 inWorldPos;

layout (location = 0) out vec4 outColor;
layout (location = 1) out vec4 outPosition;
layout (location = 2) out vec4 outNormal;
layout (location = 3) out vec4 outAlbedo;

/* layout (constant_id = 0) */ const float NEAR_PLANE = 0.1f;
/* layout (constant_id = 1) */ const float FAR_PLANE = 256.0f;

float linearDepth(float depth)
{
	float z = depth * 2.0f - 1.0f;
	return (2.0f * NEAR_PLANE * FAR_PLANE) / (FAR_PLANE + NEAR_PLANE - z * (FAR_PLANE - NEAR_PLANE));
}

void main()
{
	outPosition = vec4(inWorldPos, 1.0);

	vec3 N = normalize(inNormal);
	N.y = -N.y;
	outNormal = vec4(N, 1.0);

	outAlbedo.rgb = inColor;

	// Store linearized depth in alpha component
	outAlbedo.a = linearDepth(gl_FragCoord.z);

	// Write color attachments to avoid undefined behaviour (validation error)
	outColor = vec4(0.0);
}
"
    }
}

pub mod vs_composition {
    vulkano_shaders::shader! {
            ty: "vertex",
            src: "
#version 450

layout (location = 0) out vec2 outUV;

out gl_PerVertex
{
	vec4 gl_Position;
};

void main()
{
	outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	gl_Position = vec4(outUV * 2.0f - 1.0f, 0.0f, 1.0f);
}
"
    }
}

pub mod fs_composition {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450


layout (input_attachment_index = 0, binding = 0) uniform subpassInput samplerPosition;
layout (input_attachment_index = 1, binding = 1) uniform subpassInput samplerNormal;
layout (input_attachment_index = 2, binding = 2) uniform subpassInput samplerAlbedo;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outColor;

/* layout (constant_id = 0) */ const int NUM_LIGHTS = 64;

struct Light {
	vec4 position;
	vec3 color;
	float radius;
};

layout (binding = 3) uniform UBO
{
	vec4 viewPos;
	// Light lights[NUM_LIGHTS];
	Light lights[NUM_LIGHTS];
} ubo;


void main()
{
	// Read G-Buffer values from previous sub pass
	vec3 fragPos = subpassLoad(samplerPosition).rgb;
	vec3 normal = subpassLoad(samplerNormal).rgb;
	vec4 albedo = subpassLoad(samplerAlbedo);

	#define ambient 0.15

	// Ambient part
	vec3 fragcolor  = albedo.rgb * ambient;

	for(int i = 0; i < NUM_LIGHTS; ++i)
	{
		// Vector to light
		vec3 L = ubo.lights[i].position.xyz - fragPos;
		// Distance from light to fragment position
		float dist = length(L);

		// Viewer to fragment
		vec3 V = ubo.viewPos.xyz - fragPos;
		V = normalize(V);

		// Light to fragment
		L = normalize(L);

		// Attenuation
		float atten = ubo.lights[i].radius / (pow(dist, 2.0) + 1.0);

		// Diffuse part
		vec3 N = normalize(normal);
		float NdotL = max(0.0, dot(N, L));
		vec3 diff = ubo.lights[i].color * albedo.rgb * NdotL * atten;

		// Specular part
		// Specular map values are stored in alpha of albedo mrt
		vec3 R = reflect(-L, N);
		float NdotR = max(0.0, dot(R, V));
		//vec3 spec = ubo.lights[i].color * albedo.a * pow(NdotR, 32.0) * atten;

		fragcolor += diff;// + spec;
	}

	outColor = vec4(fragcolor, 1.0);
}
"
    }
}

pub mod vs_transparent {
    vulkano_shaders::shader! {
            ty: "vertex",
            src: "
#version 450


layout (location = 0) in vec3 position;
layout (location = 1) in vec4 color;
layout (location = 2) in vec3 normal;
layout (location = 3) in vec2 uv;

layout (binding = 0) uniform UBO
{
	mat4 projection;
	mat4 model;
	mat4 view;
} ubo;

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outUV;

void main ()
{
	outColor = color.rgb;
	outUV = uv;

	gl_Position = ubo.projection * ubo.view * ubo.model * vec4(position, 1.0);
}
"
    }
}

pub mod fs_transparent {
    vulkano_shaders::shader! {
            ty: "fragment",
            src: "
#version 450

// TODO: input_attachment_index = 1?
layout (input_attachment_index = 0, binding = 1) uniform subpassInput samplerPositionDepth;
layout (binding = 2) uniform sampler2D samplerTexture;

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 inUV;

layout (location = 0) out vec4 outColor;

/*layout (constant_id = 0)*/ const float NEAR_PLANE = 0.1f;
/*layout (constant_id = 1)*/ const float FAR_PLANE = 256.0f;

float linearDepth(float depth)
{
	float z = depth * 2.0f - 1.0f;
	return (2.0f * NEAR_PLANE * FAR_PLANE) / (FAR_PLANE + NEAR_PLANE - z * (FAR_PLANE - NEAR_PLANE));
}

void main ()
{
	// Sample depth from deferred depth buffer and discard if obscured
	float depth = subpassLoad(samplerPositionDepth).a;

	// Save the sampled texture color before discarding.
	// This is to avoid implicit derivatives in non-uniform control flow.

	vec4 sampledColor = texture(samplerTexture, inUV);

	if ((depth != 0.0) && (linearDepth(gl_FragCoord.z) > depth))
	{
		discard;
	};

	outColor = sampledColor;
}
"
    }
}
