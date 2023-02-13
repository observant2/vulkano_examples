pub mod vs_gbuffer {
    vulkano_shaders::shader! {
            ty: "vertex",
            src: "
#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec4 color;
layout (location = 2) in vec3 normal;
layout (location = 3) in vec2 uv;

layout (set = 0, binding = 0) uniform ModelViewProjection
{
	mat4 model;
	mat4 view;
	mat4 projection;
} ubo;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outUV;
layout (location = 2) out vec3 outColor;
layout (location = 3) out vec3 outPos;

void main()
{
	gl_Position = ubo.projection * ubo.view * ubo.model * vec4(position, 1.0);

	outUV = uv;

	// Vertex position in view space(!)
	outPos = vec3(ubo.view * ubo.model * vec4(position, 1.0));

	// Normal in view space
	mat3 normalMatrix = transpose(inverse(mat3(ubo.view * ubo.model)));
	outNormal = normalMatrix * normal;

	outColor = color.rgb;
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
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inColor;
layout (location = 3) in vec3 inPos;

layout (location = 0) out vec4 outColor;
layout (location = 1) out vec4 outPosition;
layout (location = 2) out vec4 outNormal;
layout (location = 3) out vec4 outAlbedo;

layout (set = 0, binding = 0) uniform ModelViewProjection
{
	mat4 model;
	mat4 view;
	mat4 projection;
} ubo;

// layout (set = 1, binding = 0) uniform sampler2D samplerColormap;

void main()
{
    outColor = vec4(1.0);
	outPosition = vec4(inPos, 1.0);
	outNormal = vec4(normalize(inNormal) * 0.5 + 0.5, 1.0);
	outAlbedo = /* texture(samplerColormap, inUV) */ vec4(inColor, 1.0);
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

layout (binding = 0) uniform sampler2D samplerPosition;
layout (binding = 1) uniform sampler2D samplerNormal;
layout (binding = 2) uniform sampler2D ssaoNoise;

layout (constant_id = 0) const int SSAO_KERNEL_SIZE = 64;
layout (constant_id = 1) const float SSAO_RADIUS = 0.5;

layout (binding = 3) uniform SsaoKernel
{
	vec4 samples[64];
} uboSSAOKernel;

layout (binding = 4) uniform Projection
{
	mat4 projection;
} ubo;

layout (location = 0) in vec2 inUV;

layout (location = 0) out float outFragColor;

void main()
{
	// Get G-Buffer values
	vec3 fragPos = texture(samplerPosition, inUV).rgb;
	vec3 normal = normalize(texture(samplerNormal, inUV).rgb * 2.0 - 1.0);

	// Get a random vector using a noise lookup
	ivec2 texDim = textureSize(samplerPosition, 0);
	ivec2 noiseDim = textureSize(ssaoNoise, 0);
	const vec2 noiseUV = vec2(float(texDim.x)/float(noiseDim.x), float(texDim.y)/(noiseDim.y)) * inUV;
	vec3 randomVec = texture(ssaoNoise, noiseUV).xyz * 2.0 - 1.0;

	// Create TBN matrix
	vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
	vec3 bitangent = cross(tangent, normal);
	mat3 TBN = mat3(tangent, bitangent, normal);

	// Calculate occlusion value
	float occlusion = 0.0f;
	// remove banding
	const float bias = 0.025f;
	for(int i = 0; i < SSAO_KERNEL_SIZE; i++)
	{
		vec3 samplePos = TBN * uboSSAOKernel.samples[i].xyz; // from tangent to view-space
		samplePos = fragPos + samplePos * SSAO_RADIUS;

		// project
		vec4 offset = vec4(samplePos, 1.0f);
		offset = ubo.projection * offset;       // from view to clip-space
		offset.xyz /= offset.w;                 // perspective divide
		offset.xyz = offset.xyz * 0.5f + 0.5f;  // transform to range 0.0..=1.0

		vec3 occluderPos = texture(samplerPosition, offset.xy).xyz;

		float rangeCheck = smoothstep(0.0f, 1.0f, SSAO_RADIUS / abs(fragPos.z - occluderPos.z));
		occlusion += (occluderPos.z >= samplePos.z + bias ? 1.0f : 0.0f) * rangeCheck;
	}

	occlusion = 1.0 - (occlusion / float(SSAO_KERNEL_SIZE));

	outFragColor = occlusion;
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
layout (binding = 5) uniform SsaoSettings
{
	int ssao;
	int ssao_only;
	int ssao_blur;
} uboParams;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

void main()
{
	vec3 fragPos = texture(samplerposition, inUV).rgb;
	vec3 normal = normalize(texture(samplerNormal, inUV).rgb * 2.0 - 1.0);
	vec4 albedo = texture(samplerAlbedo, inUV);

	float ssao = (uboParams.ssao_blur == 1) ? texture(samplerSSAOBlur, inUV).r : texture(samplerSSAO, inUV).r;

	vec3 lightPos = vec3(30, -20.0, -30);
	vec3 L = normalize(lightPos - fragPos);
	float NdotL = max(1.5, dot(normal, L));

	if (uboParams.ssao_only == 1)
	{
		outFragColor.rgb = ssao.rrr;
	}
	else
	{
		vec3 baseColor = albedo.rgb * NdotL;

		if (uboParams.ssao == 1)
		{
			outFragColor.rgb = ssao.rrr;

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
