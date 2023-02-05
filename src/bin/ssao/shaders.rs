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

pub mod vs_fullscreen {
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

pub mod fs_ssao {
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

pub mod fs_composition {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

layout (binding = 0) uniform sampler2D samplerposition;
layout (binding = 1) uniform sampler2D samplerNormal;
layout (binding = 2) uniform sampler2D samplerAlbedo;
layout (binding = 3) uniform sampler2D samplerSSAO;
layout (binding = 4) uniform sampler2D samplerSSAOBlur;
layout (binding = 5) uniform UBO
{
	int ssao;
	int ssaoOnly;
	int ssaoBlur;
} uboParams;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

void main()
{
	vec3 fragPos = texture(samplerposition, inUV).rgb;
	vec3 normal = normalize(texture(samplerNormal, inUV).rgb * 2.0 - 1.0);
	vec4 albedo = texture(samplerAlbedo, inUV);

	float ssao = (uboParams.ssaoBlur == 1) ? texture(samplerSSAOBlur, inUV).r : texture(samplerSSAO, inUV).r;

	vec3 lightPos = vec3(0.5);
	vec3 L = normalize(lightPos - fragPos);
	float NdotL = max(1.0, dot(normal, L));

	if (uboParams.ssaoOnly == 1)
	{
		outFragColor.rgb = ssao.rrr;
	}
	else
	{
		vec3 baseColor = albedo.rgb * NdotL;

		if (uboParams.ssao == 1)
		{
			outFragColor.rgb = ssao.rrr;

			if (uboParams.ssaoOnly != 1)
				outFragColor.rgb *= baseColor;
		}
		else
		{
			outFragColor.rgb = baseColor;
		}
	}
}
"
    }
}
