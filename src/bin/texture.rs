use std::collections::BTreeMap;
use std::io::Cursor;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use ktx::KtxInfo;
use nalgebra_glm::{identity, Mat4, vec3, Vec4};
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

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub normal: [f32; 3],
}

vulkano::impl_vertex!(Vertex, position, uv, normal);

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct UBO {
    model_view: Mat4,
    projection: Mat4,
    camera_pos: Vec4,
    lod_bias: f32,
}

struct Example {
    quad_vertices: Vec<Vertex>,
    quad_indices: Vec<u16>,
}

impl Example {
    fn new() -> Self {
        Self {
            quad_vertices: vec![
                Vertex {
                    position: [1.0, 1.0, 0.0],
                    uv: [1.0, 1.0],
                    normal: [0.0, 0.0, 1.0],
                },
                Vertex { position: [-1.0, 1.0, 0.0], uv: [0.0, 1.0], normal: [0.0, 0.0, 1.0] },
                Vertex { position: [-1.0, -1.0, 0.0], uv: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
                Vertex { position: [1.0, -1.0, 0.0], uv: [1.0, 0.0], normal: [0.0, 0.0, 1.0] },
            ],
            quad_indices: vec![0, 1, 2, 2, 3, 0],
        }
    }
}

mod vs {
    vulkano_shaders::shader! {
            ty: "vertex",
            src: "
#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;
layout (location = 2) in vec3 normal;

layout (binding = 0) uniform UBO
{
	mat4 model;
	mat4 projection;
	vec4 viewPos;
	float lodBias;
} ubo;

layout (location = 0) out vec2 outUV;
layout (location = 1) out float outLodBias;
layout (location = 2) out vec3 outNormal;
layout (location = 3) out vec3 outViewVec;
layout (location = 4) out vec3 outLightVec;

out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{
	outUV = uv;
	outLodBias = ubo.lodBias;

	vec3 worldPos = vec3(ubo.model * vec4(position, 1.0));

	gl_Position = ubo.projection * ubo.model * vec4(position, 1.0);

    vec4 pos = ubo.model * vec4(position, 1.0);
	outNormal = mat3(inverse(transpose(ubo.model))) * normal;
	vec3 lightPos = vec3(0.0);
	vec3 lPos = mat3(ubo.model) * lightPos.xyz;
    outLightVec = lPos - pos.xyz;
    outViewVec = ubo.viewPos.xyz - pos.xyz;
}
"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

layout (binding = 1) uniform sampler2D samplerColor;

layout (location = 0) in vec2 inUV;
layout (location = 1) in float inLodBias;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;

layout (location = 0) out vec4 outFragColor;

void main()
{
	vec4 color = texture(samplerColor, inUV, inLodBias);

	vec3 N = normalize(inNormal);
	vec3 L = normalize(inLightVec);
	vec3 V = normalize(inViewVec);
	vec3 R = reflect(-L, N);
	vec3 diffuse = max(dot(N, L), 0.0) * vec3(1.0);
	float specular = pow(max(dot(R, V), 0.0), 16.0) * color.a;

	outFragColor = vec4(diffuse * color.rgb + specular, 1.0);
}
"
    }
}

fn get_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        .with_pipeline_layout(device.clone(), PipelineLayout::new(device.clone(), PipelineLayoutCreateInfo {
            set_layouts: vec![
                DescriptorSetLayout::new(device, DescriptorSetLayoutCreateInfo {
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

pub fn main() {
    let (mut app, event_loop) = App::new("texture");

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
    let mut example = Example::new();

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        memory_allocator.as_ref(),
        BufferUsage::VERTEX_BUFFER,
        false,
        example.quad_vertices,
    )
        .expect("failed to create buffer");

    let index_buffer = CpuAccessibleBuffer::from_iter(
        memory_allocator.as_ref(),
        BufferUsage::INDEX_BUFFER,
        false,
        example.quad_indices,
    ).expect("failed to create index buffer");

    // Create buffers for descriptorset 0, binding 0

    let ubo_buffer = CpuAccessibleBuffer::from_data(
        memory_allocator.as_ref(),
        BufferUsage::UNIFORM_BUFFER,
        false,
        UBO {
            model_view: identity(),
            projection: identity(),
            camera_pos: Vec4::identity(),
            lod_bias: 0.0,
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
    let mut pipeline = get_pipeline(
        app.device.clone(),
        vs_shader.clone(),
        fs_shader.clone(),
        render_pass.clone(),
        viewport.clone(),
    );

    let layout = pipeline.layout().set_layouts().get(0).unwrap();

    let descriptor_allocator = StandardDescriptorSetAllocator::new(app.device.clone());

    // Create a separate command buffer for uploading textures to gpu memory

    let mut uploads = AutoCommandBufferBuilder::primary(
        &app.allocator_command_buffer,
        app.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
        .unwrap();

    // The textures will be bound to descriptorset 0, binding 1

    let texture = {
        let bytes = include_bytes!("../../data/textures/metalplate01_rgba.ktx").to_vec();
        let cursor = Cursor::new(bytes);
        let decoder = ktx::Decoder::new(cursor).unwrap();
        let mips = decoder.mipmap_levels();
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
            MipmapsCount::Specific(mips),
            Format::R8G8B8A8_UNORM,
            &mut uploads,
        )
            .unwrap()
    };
    let sampler = Sampler::new(app.device.clone(), SamplerCreateInfo::simple_repeat_linear()).unwrap();

    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_allocator,
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, ubo_buffer.clone()),
            WriteDescriptorSet::image_view_sampler(1, ImageView::new_default(texture.clone()).unwrap(), sampler.clone())
        ],
    ).unwrap();

    let _ = uploads
        .build()
        .unwrap()
        .execute(app.queue.clone())
        .unwrap()
        .then_signal_fence_and_flush();

    let mut command_buffers = get_command_buffers(&app, &pipeline, &framebuffers, &vertex_buffer, &index_buffer, &descriptor_set);

    let mut recreate_swapchain = true;

    let mut last_frame = Instant::now();

    let mut previous_frame_end = Some(sync::now(app.device.clone()).boxed());

    let mut camera = Camera::new(vec3(0.0, 0.0, -2.5), aspect_ratio, f32::to_radians(60.0), 0.01, 256.0);

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
                    pipeline = get_pipeline(
                        app.device.clone(),
                        vs_shader.clone(),
                        fs_shader.clone(),
                        render_pass.clone(),
                        viewport.clone(),
                    );

                    let aspect_ratio =
                        app.swapchain.image_extent()[0] as f32 / app.swapchain.image_extent()[1] as f32;
                    camera.set_perspective(aspect_ratio, f32::to_radians(60.0), 0.01, 512.0);

                    command_buffers = get_command_buffers(&app, &pipeline, &framebuffers, &vertex_buffer, &index_buffer, &descriptor_set);

                    recreate_swapchain = false;
                }

                {
                    if let Ok(mut ubo) = ubo_buffer.write() {
                        ubo.model_view = camera.get_view_matrix();
                        ubo.projection = camera.get_perspective_matrix();
                        ubo.camera_pos = Vec4::from_data(camera.position.xyz().push(1.0).data);
                        ubo.camera_pos.x *= -1.0;
                        ubo.camera_pos.z *= -1.0;
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
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    vertex_buffer: &Arc<CpuAccessibleBuffer<[Vertex]>>,
    index_buffer: &Arc<CpuAccessibleBuffer<[u16]>>,
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
                ).unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                // bind one vertex buffer for both cubes (same geometry)
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .bind_index_buffer(index_buffer.clone());

            builder
                // bind descriptorset containing matrices and texture for quad
                .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline.layout().clone(), 0,
                                      vec![descriptor_set.clone()])
                // draw quad
                .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0).unwrap();

            builder.end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}
