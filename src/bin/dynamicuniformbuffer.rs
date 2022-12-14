use std::collections::btree_map::BTreeMap;
use std::f32::consts::PI;
use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use nalgebra_glm::{identity, Mat4, rotate, translate, vec3, Vec3};
use rand::Rng;
use vulkano::{swapchain, sync, DeviceSize};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents};
use vulkano::descriptor_set::{DescriptorSet, PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType};
use vulkano::device::Device;
use vulkano::format::{ClearValue, Format};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};
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

const INSTANCES: usize = 125;

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

vulkano::impl_vertex!(Vertex, position, color);

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct ViewProjection {
    view: Mat4,
    projection: Mat4,
}

struct Example {
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
    rotations: [Vec3; INSTANCES],
    rotation_speeds: [Vec3; INSTANCES],
}

impl Example {
    fn new() -> Self {
        let mut rand = rand::thread_rng();
        let r1 = 0.0..1.0;
        let r2 = -1.0..1.0;
        let mut rotations = [vec3(0.0, 0.0, 0.0); INSTANCES];
        let mut rotation_speeds = [vec3(0.0, 0.0, 0.0); INSTANCES];

        for i in 0..INSTANCES {
            rotations[i] = vec3(rand.gen_range(r1.clone()), rand.gen_range(r1.clone()), rand.gen_range(r1.clone())) * PI * 2.0;
            rotation_speeds[i] = vec3(rand.gen_range(r2.clone()), rand.gen_range(r2.clone()), rand.gen_range(r2.clone())) * 0.3;
        }

        Example {
            vertices: Vec::from([
                Vertex { position: [-1.0, -1.0, 1.0], color: [1.0, 0.0, 0.0] },
                Vertex { position: [1.0, -1.0, 1.0], color: [0.0, 1.0, 0.0] },
                Vertex { position: [1.0, 1.0, 1.0], color: [0.0, 0.0, 1.0] },
                Vertex { position: [-1.0, 1.0, 1.0], color: [0.0, 0.0, 0.0] },
                Vertex { position: [-1.0, -1.0, -1.0], color: [1.0, 0.0, 0.0] },
                Vertex { position: [1.0, -1.0, -1.0], color: [0.0, 1.0, 0.0] },
                Vertex { position: [1.0, 1.0, -1.0], color: [0.0, 0.0, 1.0] },
                Vertex { position: [-1.0, 1.0, -1.0], color: [0.0, 0.0, 0.0] },
            ]),

            indices: vec![
                0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7, 4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4, 3, 2, 6, 6, 7, 3,
            ],
            rotations,
            rotation_speeds,
        }
    }
}

mod vs {
    vulkano_shaders::shader! {
            ty: "vertex",
            src: "
#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;

layout (set = 0, binding = 0) uniform UboView
{
	mat4 view;
	mat4 projection;
} uboView;

layout (set = 0, binding = 1) uniform UboInstance
{
    mat4 model;
} uboInstance;

layout (location = 0) out vec3 outColor;

void main()
{
	outColor = color;
    mat4 modelView = uboView.view * uboInstance.model;
	gl_Position =  uboView.projection * modelView * vec4(position, 1.0);
}
"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

layout (location = 0) in vec3 inColor;

layout (location = 0) out vec4 outFragColor;

void main()
{
	outFragColor = vec4(inColor, 1.0);
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
                            stages: ShaderStages::VERTEX,
                            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBufferDynamic)
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
    let (mut app, event_loop) = App::new("dynamic uniform buffers");

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

    // Create vertex buffers

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        memory_allocator.as_ref(),
        BufferUsage::VERTEX_BUFFER,
        false,
        example.vertices,
    )
        .expect("failed to create buffer");

    let index_buffer = CpuAccessibleBuffer::from_iter(
        memory_allocator.as_ref(),
        BufferUsage::INDEX_BUFFER,
        false,
        example.indices,
    ).expect("failed to create index buffer");


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

    let min_ubo_alignment = app.device
        .physical_device()
        .properties()
        .min_uniform_buffer_offset_alignment as usize;
    println!(
        "Minimum uniform buffer offset alignment: {}",
        min_ubo_alignment
    );

    // Calculate required alignment based on minimum device offset alignment
    let mut dynamic_alignment = size_of::<Mat4>();
    if min_ubo_alignment > 0 {
        dynamic_alignment = (dynamic_alignment + min_ubo_alignment - 1) & !(min_ubo_alignment - 1);
    }

    let buffer_size = INSTANCES * dynamic_alignment;

    // Create empty buffer of exactly as many bytes as calculated
    let aligned_data = vec![0u8; buffer_size];

    // This buffer will be used later to store the model matrices of the cubes
    let dynamic_model_buffer = CpuAccessibleBuffer::from_iter(
        memory_allocator.as_ref(),
        BufferUsage::UNIFORM_BUFFER,
        false,
        aligned_data.into_iter(),
    ).unwrap();


    let view_projection_buffer = CpuAccessibleBuffer::from_data(
        memory_allocator.as_ref(),
        BufferUsage::UNIFORM_BUFFER,
        false,
        ViewProjection {
            projection: identity(),
            view: identity(),
        },
    ).unwrap();

    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    let descriptor_allocator = StandardDescriptorSetAllocator::new(app.device.clone());

    let view_projection_set = PersistentDescriptorSet::new(
        &descriptor_allocator,
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, view_projection_buffer.clone()),
            // dynamic uniform buffers need a buffer with range.
            // The range specifies the byte index of the first element and the size of one element in bytes.
            // It does not specify the range of the whole buffer!
            WriteDescriptorSet::buffer_with_range(1, dynamic_model_buffer.clone(), 0..size_of::<Mat4>() as DeviceSize),
        ],
    ).unwrap();


    let mut command_buffers = get_command_buffers(&app, &pipeline, &framebuffers, &vertex_buffer, &index_buffer, &view_projection_set, dynamic_alignment);

    let mut recreate_swapchain = true;

    let mut last_frame = Instant::now();

    let mut previous_frame_end = Some(sync::now(app.device.clone()).boxed());

    let mut camera = Camera::new(vec3(0.0, 0.0, -30.0), aspect_ratio, f32::to_radians(60.0), 0.01, 256.0);

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

                {
                    // Update rotations and matrices for every instance
                    let dim = (INSTANCES as f32).powf(1.0 / 3.0) as u32; // this is an integer, because INSTANCES is a power of three
                    let offset = Vec3::from_element(5.0);

                    let dmb = dynamic_model_buffer.write();

                    if let Ok(..) = dmb {
                        let mut dmb = dmb.unwrap();
                        for x in 0..dim
                        {
                            for y in 0..dim
                            {
                                for z in 0..dim
                                {
                                    let index = (x * dim * dim + y * dim + z) as usize;

                                    // reinterpret the raw bytes of the dynamic_model_buffer as a slice of Mat4
                                    let model_mat: &mut [Mat4] = bytemuck::cast_slice_mut(&mut dmb[index * dynamic_alignment..(index * dynamic_alignment + dynamic_alignment)]);

                                    // Update rotations
                                    example.rotations[index] += 0.16 * example.rotation_speeds[index];

                                    // Update matrices
                                    let pos = vec3(-((dim as f32 * offset.x) / 2.0) + offset.x / 2.0 + x as f32 * offset.x,
                                                   -((dim as f32 * offset.y) / 2.0) + offset.y / 2.0 + y as f32 * offset.y,
                                                   -((dim as f32 * offset.z) / 2.0) + offset.z / 2.0 + z as f32 * offset.z);
                                    *model_mat[0] = *translate(&identity(), &pos);
                                    *model_mat[0] = *rotate(&model_mat[0], example.rotations[index].x, &vec3(1.0, 1.0, 0.0));
                                    *model_mat[0] = *rotate(&model_mat[0], example.rotations[index].y, &vec3(0.0, 1.0, 0.0));
                                    *model_mat[0] = *rotate(&model_mat[0], example.rotations[index].z, &vec3(0.0, 0.0, 1.0));
                                }
                            }
                        }
                    }
                }

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

                    command_buffers = get_command_buffers(&app, &pipeline, &framebuffers, &vertex_buffer, &index_buffer, &view_projection_set, dynamic_alignment);

                    recreate_swapchain = false;
                }

                {
                    let view_projection = view_projection_buffer.write();

                    if view_projection.is_ok() {
                        let mut view_projection = view_projection.unwrap();

                        view_projection.view = camera.get_view_matrix();
                        view_projection.projection = camera.get_perspective_matrix();
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
    dynamic_set: &Arc<PersistentDescriptorSet>,
    dynamic_align: usize,
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
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .bind_index_buffer(index_buffer.clone());
            for i in 0..INSTANCES {
                let offset = (i * dynamic_align) as u32;
                builder
                    // bind descriptorset at offset for current instance, offset again in bytes!
                    .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline.layout().clone(), 0,
                                          vec![dynamic_set.clone().offsets([offset])],
                    )
                    // draw cube
                    .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0).unwrap();
            }

            builder.end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}
