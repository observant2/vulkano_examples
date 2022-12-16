use std::collections::btree_map::BTreeMap;
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
use vulkano::image::{AttachmentImage, ImageAccess, SwapchainImage};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
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
use vulkano_examples::gltf_loader::Model;

const INSTANCES: usize = 125;

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub normal: [f32; 3],
}

vulkano::impl_vertex!(Vertex, position, color, normal);

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct ViewProjection {
    view: Mat4,
    projection: Mat4,
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
layout (location = 1) in vec3 color;
layout (location = 2) in vec3 normal;

layout (set = 0, binding = 0) uniform ViewProjection
{
	mat4 view;
	mat4 projection;
} ubo;

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec3 outNormal;

void main()
{
	outColor = color;
	gl_Position =  ubo.projection * ubo.view * vec4(position, 1.0);
    outNormal = normal;
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
layout (location = 1) in vec3 inNormal;

layout (location = 0) out vec4 outFragColor;

void main()
{
	outFragColor = vec4(inColor, 1.0);
}
"
    }
}


fn get_framebuffers(memory_allocator: &Arc<StandardMemoryAllocator>, images: &[Arc<SwapchainImage>], render_pass: &Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let color = ImageView::new_default(image.clone()).unwrap();
            // let color_write = ImageView::new_default(
            //     AttachmentImage::input_attachment(memory_allocator, image.dimensions().width_height(), image.format()).unwrap()
            // ).unwrap();
            let depth_buffer = ImageView::new_default(
                AttachmentImage::transient(memory_allocator, image.dimensions().width_height(), Format::D24_UNORM_S8_UINT).unwrap()
            ).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![color /*, color_write */, depth_buffer],
                    ..Default::default()
                },
            )
                .unwrap()
        })
        .collect::<Vec<_>>()
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
                        })
                    ]),
                    ..DescriptorSetLayoutCreateInfo::default()
                }).unwrap()
            ],
            ..PipelineLayoutCreateInfo::default()
        }).unwrap())
        .unwrap()
}

pub fn main() {
    let (mut app, event_loop) = App::new("inputattachments");

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(app.device.clone()));

    let color_format = app.swapchain.image_format();

    let render_pass = vulkano::ordered_passes_renderpass!(
        app.device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: color_format,
                samples: 1,
            },
            // color_write: {
            //     load: Clear,
            //     store: DontCare,
            //     format: color_format,
            //     samples: 1,
            // },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D24_UNORM_S8_UINT,
                samples: 1,
            }
        },
        passes: [{
            color: [color],
            depth_stencil: {depth},
            input: []
        //},
        // {
        //     color: [color],
        //     depth_stencil: {},
        //     input: [color_write]
         }]
    )
        .unwrap();

    let mut framebuffers = get_framebuffers(&memory_allocator, &app.swapchain_images, &render_pass);

    let aspect_ratio =
        app.swapchain.image_extent()[0] as f32 / app.swapchain.image_extent()[1] as f32;

    let meshes_from_file = Model::load("./data/models/treasure_smooth.gltf").meshes;

    let mut scene_objects = vec![];

    for mesh in meshes_from_file {
        for primitive in mesh.primitives {
            let vertices: Vec<Vertex> = primitive.vertices.iter()
                .zip(primitive.normals.iter())
                .zip(primitive.colors.iter())
                .map(|((v, n), c)| Vertex {
                    position: *v,
                    normal: *n,
                    color: (&c[..3]).try_into().unwrap(),
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
                primitive.indices,
            ).expect("failed to create index buffer");

            scene_objects.push(SceneObject {
                vertex_buffer,
                index_buffer,
            })
        }
    }

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
        ],
    ).unwrap();


    let mut command_buffers = get_command_buffers(&app, &pipeline, &framebuffers, &scene_objects, &view_projection_set);

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

                    command_buffers = get_command_buffers(&app, &pipeline, &framebuffers, &scene_objects, &view_projection_set);

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
    scene_objects: &Vec<SceneObject>,
    view_projection_set: &Arc<PersistentDescriptorSet>,
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
                .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline.layout().clone(), 0, vec![view_projection_set.clone()]);
            for scene_object in scene_objects {
                builder
                    .bind_vertex_buffers(0, scene_object.vertex_buffer.clone())
                    .bind_index_buffer(scene_object.index_buffer.clone())
                    .draw_indexed(scene_object.index_buffer.len() as u32, 1, 0, 0, 0).unwrap();
            }

            builder.end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}
