use std::collections::btree_map::BTreeMap;
use std::io::Cursor;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use ktx::KtxInfo;
use nalgebra_glm::{identity, Mat4, rotate, scale, translate, vec3, Vec3, Vec4, vec4};
use vulkano::{swapchain, sync};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassContents};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType};
use vulkano::device::Device;
use vulkano::format::{ClearValue, Format};
use vulkano::image::{ImageDimensions, ImmutableImage, MipmapsCount};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};
use vulkano::sampler::{Sampler, SamplerCreateInfo};
use vulkano::shader::{ShaderModule, ShaderStages};
use vulkano::swapchain::{
    AcquireError, SwapchainCreateInfo, SwapchainCreationError, SwapchainPresentInfo,
};
use vulkano::sync::{FlushError, GpuFuture};
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::window::Window;

use vulkano_examples::App;
use vulkano_examples::camera::Camera;
use vulkano_examples::gltf_loader::{Mesh, Model};

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}
vulkano::impl_vertex!(Vertex, position, normal, uv, color);

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct UBO {
    view: Mat4,
    projection: Mat4,
    light_pos: Vec4,
}

struct SceneObject {
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    index_buffer: Arc<CpuAccessibleBuffer<[u16]>>,
}

mod vs {
    vulkano_shaders::shader! {
            ty: "vertex",
            src: "
#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;
layout (location = 3) in vec4 color;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outUV;
layout (location = 2) out vec3 outColor;
layout (location = 3) out vec3 outViewVec;
layout (location = 4) out vec3 outLightVec;

layout (set = 0, binding = 0) uniform UBO
{
	mat4 view;
	mat4 projection;
    vec4 light_pos;
} ubo;

void main()
{
	outUV = uv;
    outColor = color.xyz;
	gl_Position = ubo.projection * ubo.view * vec4(position, 1.0);

	vec4 pos = ubo.view * vec4(position, 1.0);
	outNormal = mat3(ubo.view) * normal;
	vec3 lPos = mat3(ubo.view) * ubo.light_pos.xyz;
	outLightVec = lPos - pos.xyz;
	outViewVec = -pos.xyz;
}
"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

layout (binding = 1) uniform sampler2D samplerColormap;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inColor;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;

layout (location = 0) out vec4 outFragColor;

// We use this constant to control the flow of the shader depending on the
// lighting model selected at pipeline creation time
layout (constant_id = 0) const int LIGHTING_MODEL = 0;
// Parameter for the toon shading part of the shader
layout (constant_id = 1) const float PARAM_TOON_DESATURATION = 0.0f;

void main()
{
	switch (LIGHTING_MODEL) {
		case 0: // Phong
		{
			vec3 ambient = inColor * vec3(0.25);
			vec3 N = normalize(inNormal);
			vec3 L = normalize(inLightVec);
			vec3 V = normalize(inViewVec);
			vec3 R = reflect(-L, N);
			vec3 diffuse = max(dot(N, L), 0.0) * inColor;
			vec3 specular = pow(max(dot(R, V), 0.0), 32.0) * vec3(0.75);
			outFragColor = vec4(ambient + diffuse * 1.75 + specular, 1.0);
			break;
		}
		case 1: // Toon
		{

			vec3 N = normalize(inNormal);
			vec3 L = normalize(inLightVec);
			float intensity = dot(N,L);
			vec3 color;
			if (intensity > 0.98)
				color = inColor * 1.5;
			else if  (intensity > 0.9)
				color = inColor * 1.0;
			else if (intensity > 0.5)
				color = inColor * 0.6;
			else if (intensity > 0.25)
				color = inColor * 0.4;
			else
				color = inColor * 0.2;
			// Desaturate a bit
			color = vec3(mix(color, vec3(dot(vec3(0.2126,0.7152,0.0722), color)), PARAM_TOON_DESATURATION));
			outFragColor.rgb = color;
			break;
		}
		case 2: // Textured
		{
			vec4 color = texture(samplerColormap, inUV).rrra;
			vec3 ambient = color.rgb * vec3(0.25) * inColor;
			vec3 N = normalize(inNormal);
			vec3 L = normalize(inLightVec);
			vec3 V = normalize(inViewVec);
			vec3 R = reflect(-L, N);
			vec3 diffuse = max(dot(N, L), 0.0) * color.rgb;
			float specular = pow(max(dot(R, V), 0.0), 32.0) * color.a;
			outFragColor = vec4(ambient + diffuse + vec3(specular), 1.0);
			break;
		}
	}
}
"
    }
}

enum Specialization {
    Phong,
    Toon,
    Textured,
}

fn get_pipeline(
    device: &Arc<Device>,
    vs: &Arc<ShaderModule>,
    fs: &Arc<ShaderModule>,
    render_pass: &Arc<RenderPass>,
    viewport: &Viewport,
    specialization: Specialization,
) -> Arc<GraphicsPipeline> {
    let specialization_constants =
        match specialization {
            Specialization::Phong =>
                fs::SpecializationConstants {
                    LIGHTING_MODEL: 0,
                    PARAM_TOON_DESATURATION: 0.5,
                },
            Specialization::Toon =>
                fs::SpecializationConstants {
                    LIGHTING_MODEL: 1,
                    PARAM_TOON_DESATURATION: 0.5,
                },
            Specialization::Textured =>
                fs::SpecializationConstants {
                    LIGHTING_MODEL: 2,
                    PARAM_TOON_DESATURATION: 0.5,
                },
        };

    GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), specialization_constants /* <- */)
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .with_pipeline_layout(device.clone(), PipelineLayout::new(device.clone(), PipelineLayoutCreateInfo {
            set_layouts: vec![
                DescriptorSetLayout::new(device.clone(), DescriptorSetLayoutCreateInfo {
                    bindings: BTreeMap::from([
                        (0, DescriptorSetLayoutBinding {
                            stages: ShaderStages::VERTEX,
                            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
                        }),
                        (1, DescriptorSetLayoutBinding {
                            stages: ShaderStages::FRAGMENT,
                            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
                        }),
                    ]),
                    ..DescriptorSetLayoutCreateInfo::default()
                }).unwrap()
            ],
            ..PipelineLayoutCreateInfo::default()
        }).unwrap())
        .unwrap()
}

fn get_pipelines(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> [Arc<GraphicsPipeline>; 3] {
    [
        get_pipeline(&device, &vs, &fs, &render_pass, &viewport, Specialization::Phong),
        get_pipeline(&device, &vs, &fs, &render_pass, &viewport, Specialization::Toon),
        get_pipeline(&device, &vs, &fs, &render_pass, &viewport, Specialization::Textured),
    ]
}

pub fn main() {
    let (mut app, event_loop) = App::new("specializationconstants");

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(app.device.clone()));

    let render_pass = vulkano::single_pass_renderpass!(
        app.device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: app.swapchain.image_format(),
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D24_UNORM_S8_UINT,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    )
        .unwrap();

    let mut framebuffers = app.get_framebuffers(&memory_allocator, &app.swapchain_images, &render_pass);

    let aspect_ratio =
        app.swapchain.image_extent()[0] as f32 / app.swapchain.image_extent()[1] as f32;

    let meshes_from_file = Model::load("./assets/models/color_teapot_spheres.gltf").meshes;

    let mut scene_objects = vec![];

    // Create buffers for descriptorset 0, binding 0

    let ubo_buffer = CpuAccessibleBuffer::from_data(
        memory_allocator.as_ref(),
        BufferUsage::UNIFORM_BUFFER,
        false,
        UBO {
            view: identity(),
            projection: identity(),
            light_pos: vec4(0.0, 2.0, 1.0, 0.0),
        },
    ).unwrap();

    // Create pipeline

    let window = app.surface.object().unwrap().downcast_ref::<Window>().unwrap();
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: window.inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let vs_shader = vs::load(app.device.clone()).unwrap();
    let fs_shader = fs::load(app.device.clone()).unwrap();
    let mut pipelines = get_pipelines(
        app.device.clone(),
        vs_shader.clone(),
        fs_shader.clone(),
        render_pass.clone(),
        viewport.clone(),
    );

    let layout = pipelines[0].layout().set_layouts().get(0).unwrap();

    let mut uploads = AutoCommandBufferBuilder::primary(
        &app.allocator_command_buffer,
        app.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
        .unwrap();

    let metal_texture = {
        let metal_texture = include_bytes!("../../assets/textures/metalplate_nomips_rgba.ktx").to_vec();
        let cursor = Cursor::new(metal_texture);
        let decoder = ktx::Decoder::new(cursor).unwrap();
        let width = decoder.pixel_width();
        let height = decoder.pixel_height();
        let image_data = decoder.read_textures().next().unwrap();
        let dimensions = ImageDimensions::Dim2d {
            width,
            height,
            array_layers: 1,
        };
        ImmutableImage::from_iter(
            &memory_allocator,
            image_data,
            dimensions,
            MipmapsCount::Log2,
            Format::R8G8B8A8_SRGB,
            &mut uploads,
        )
            .unwrap()
    };

    let sampler = Sampler::new(app.device.clone(), SamplerCreateInfo::simple_repeat_linear()).unwrap();

    let descriptor_allocator = StandardDescriptorSetAllocator::new(app.device.clone());

    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_allocator,
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, ubo_buffer.clone()),
            WriteDescriptorSet::image_view_sampler(1, ImageView::new_default(metal_texture.clone()).unwrap(), sampler.clone())
        ],
    ).unwrap();


    for mesh in meshes_from_file {
        let vertices: Vec<Vertex> = mesh.vertices.iter()
            .zip(mesh.tex_coords.iter())
            .zip(mesh.normals.iter())
            .zip(mesh.colors.iter())
            .map(|(((v, t), n), c)| Vertex {
                position: *v,
                normal: *n,
                uv: *t,
                color: *c,
            }).collect();

        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            memory_allocator.as_ref(),
            BufferUsage::VERTEX_BUFFER,
            false,
            vertices,
        )
            .expect("failed to create buffer");

        let index_buffer = CpuAccessibleBuffer::from_iter(
            memory_allocator.as_ref(),
            BufferUsage::INDEX_BUFFER,
            false,
            mesh.indices,
        ).expect("failed to create index buffer");

        scene_objects.push(SceneObject {
            vertex_buffer,
            index_buffer,
        })
    }

    let _ = uploads
        .build()
        .unwrap()
        .execute(app.queue.clone())
        .unwrap()
        .then_signal_fence_and_flush();

    let mut command_buffers = get_command_buffers(&app, &pipelines, &framebuffers, &scene_objects, &descriptor_set);

    let mut recreate_swapchain = true;

    let mut last_frame = Instant::now();

    let mut previous_frame_end = Some(sync::now(app.device.clone()).boxed());

    let mut camera = Camera::new(vec3(0.0, 0.0, -2.0), aspect_ratio, f32::to_radians(60.0), 0.01, 512.0);

    event_loop.run(move |event, _, control_flow| {
        camera.handle_input(&event);

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::ExitWithCode(0);
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::MainEventsCleared => {
                camera.update_view_matrix();
            }
            Event::RedrawRequested(..) => {
                let elapsed = last_frame.elapsed().as_millis();
                if elapsed < (1000.0 / 60.0) as u128 {
                    return;
                } else {
                    last_frame = Instant::now();
                }

                previous_frame_end.as_mut().unwrap().cleanup_finished();

                let window = app.surface.object().unwrap().downcast_ref::<Window>().unwrap();

                if window.inner_size().width == 0 || window.inner_size().height == 0 {
                    return;
                }

                if recreate_swapchain {
                    let (new_swapchain, new_images) =
                        match app.swapchain.recreate(SwapchainCreateInfo {
                            image_extent: window.inner_size().into(),
                            ..app.swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("failed to recreate swapchain: {:?}", e),
                        };
                    app.swapchain = new_swapchain;
                    framebuffers = app.get_framebuffers(&memory_allocator, &new_images, &render_pass);

                    viewport.dimensions = window.inner_size().into();
                    pipelines = get_pipelines(
                        app.device.clone(),
                        vs_shader.clone(),
                        fs_shader.clone(),
                        render_pass.clone(),
                        viewport.clone(),
                    );

                    let [width, height] = app.swapchain.image_extent();
                    camera.set_perspective(width as f32 / 3.0 / height as f32, f32::to_radians(60.0), 0.01, 512.0);

                    command_buffers = get_command_buffers(&app, &pipelines, &framebuffers, &scene_objects, &descriptor_set);

                    recreate_swapchain = false;
                }

                {
                    if let Ok(mut ubo) = ubo_buffer.write() {
                        ubo.projection = camera.get_perspective_matrix();
                        ubo.view = camera.get_view_matrix();
                    }
                }

                let (image_i, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(app.swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {:?}", e),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(app.queue.clone(), command_buffers[image_i as usize].clone())
                    .unwrap()
                    .then_swapchain_present(
                        app.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(app.swapchain.clone(), image_i),
                    )
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(app.device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(app.device.clone()).boxed());
                    }
                }
            }
            _ => {
                let window = app.surface.object().unwrap().downcast_ref::<Window>().unwrap();
                window.request_redraw();
            }
        }
    });
}

fn get_command_buffers(
    app: &App,
    pipelines: &[Arc<GraphicsPipeline>; 3],
    framebuffers: &[Arc<Framebuffer>],
    scene_objects: &Vec<SceneObject>,
    descriptor_set: &Arc<PersistentDescriptorSet>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                app.allocator_command_buffer.as_ref(),
                app.queue_family_index,
                CommandBufferUsage::MultipleSubmit,
            )
                .unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![
                            Some([0.1, 0.1, 0.1, 1.0].into()),
                            Some(ClearValue::DepthStencil((1.0, 0))),
                        ],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassContents::Inline,
                ).unwrap();

            let window = app.surface.object().unwrap().downcast_ref::<Window>().unwrap();

            let mut viewport = Viewport {
                origin: [0.0, 0.0],
                dimensions: [window.inner_size().width as f32 / 3.0, window.inner_size().height as f32],
                depth_range: 0.0..1.0,
            };

            for i in 0..3 {
                viewport.origin[0] = (window.inner_size().width as f32 / 3.0) * i as f32;
                builder.set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(pipelines[i].clone());

                for scene_object in scene_objects {
                    builder
                        .bind_vertex_buffers(0, scene_object.vertex_buffer.clone())
                        .bind_index_buffer(scene_object.index_buffer.clone())
                        .bind_descriptor_sets(PipelineBindPoint::Graphics, pipelines[i].layout().clone(), 0, vec![descriptor_set.clone()])
                        .draw_indexed(scene_object.index_buffer.len() as u32, 1, 0, 0, 0).unwrap();
                }
            }

            builder.end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}
