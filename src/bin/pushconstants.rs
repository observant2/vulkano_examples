use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use gltf::{Document, Semantic};
use gltf::accessor::Dimensions;
use gltf::buffer::Data;
use gltf::json::accessor::ComponentType;
use nalgebra_glm::{identity, Mat4, perspective, vec3, Vec4, vec4};
use rand::Rng;
use vulkano_examples::{App};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents};
use vulkano::device::Device;

use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{
    AcquireError, SwapchainCreateInfo, SwapchainCreationError, SwapchainPresentInfo,
};
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::{swapchain, sync};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::format::{ClearValue, Format};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use winit::event::{ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event::DeviceEvent::{Button, MouseMotion};
use winit::event::WindowEvent::MouseWheel;
use winit::event_loop::{ControlFlow};
use winit::window::{Window};
use vulkano_examples::camera::Camera;

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct UBO {
    projection: Mat4,
    model: Mat4,
    view: Mat4,
}

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct SpherePushConstantData {
    color: Vec4,
    position: Vec4,
}


mod vs {
    vulkano_shaders::shader! {
            ty: "vertex",
            src: "
#version 450

layout (location = 0) in vec3 position;

layout (binding = 0) uniform UBO
{
	mat4 projection;
	mat4 model;
	mat4 view;
} ubo;

layout(push_constant) uniform PushConsts {
	vec4 color;
	vec4 position;
} pushConsts;

layout (location = 0) out vec3 outColor;

void main()
{
	outColor = pushConsts.color.rgb;
	vec3 locPos = vec3(ubo.model * vec4(position, 1.0));
	vec3 worldPos = locPos + pushConsts.position.xyz;
	gl_Position =  ubo.projection * ubo.view * vec4(worldPos, 1.0);
    gl_PointSize = 2;
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
	outFragColor.rgb = inColor;
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
        .build(device)
        .unwrap()
}

struct Example {
    spheres: [SpherePushConstantData; 16],

    model: Document,
    buffers: Vec<Data>,
}

impl Example {
    fn new() -> Self {
        let (model, buffers, _) = gltf::import("./assets/sphere.gltf").expect("couldn't load model");

        let mut ex = Example {
            spheres: Default::default(),
            model,
            buffers,
        };

        let mut rnd = rand::thread_rng();

        let mut random_num = move || { rnd.gen_range(0.0..1.0) };

        // Setup random colors and fixed positions for every spheres in the scene
        for i in 0..ex.spheres.len() {
            ex.spheres[i].color = Vec4::new(random_num(), random_num(), random_num(), 1.0);
            let rad = f32::to_radians(i as f32 * 360.0 / ex.spheres.len() as f32);
            ex.spheres[i].position = Vec4::new(f32::sin(rad), f32::cos(rad), 0.0, 1.0);
        }

        ex
    }
}

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
pub struct Vertex {
    pub position: [f32; 3],
}
vulkano::impl_vertex!(Vertex, position);

pub fn main() {
    let (mut app, event_loop) = App::new("pushconstants");

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(app.device.clone()));

    let vs_shader = vs::load(app.device.clone()).unwrap();
    let fs_shader = fs::load(app.device.clone()).unwrap();

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
                format: Format::D32_SFLOAT_S8_UINT,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    )
        .unwrap();

    let framebuffers = app.get_framebuffers(&memory_allocator, &app.swapchain_images, &render_pass);

    let example = Example::new();

    let mut vertices = Vec::<Vertex>::new();
    let mut indices = Vec::<u16>::new();

    for mesh in example.model.meshes() {
        for primitive in mesh.primitives() {
            println!("primitive nr.:{:?}", primitive.index());
            for attr in primitive.attributes() {
                if attr.0 == Semantic::Positions {
                    if attr.1.dimensions() != Dimensions::Vec3 {
                        panic!("Positions of gltf need to be vec3");
                    };

                    if attr.1.data_type() == ComponentType::F32 {
                        let accessor = attr.1;
                        let buffer_view = accessor.view().unwrap();
                        let contents: &[f32] = bytemuck::cast_slice(&example.buffers[0][buffer_view.offset()..(buffer_view.offset() + buffer_view.length())]);
                        for i in (0..accessor.count() * 3).step_by(3) {
                            vertices.push(Vertex {
                                position: [contents[i], contents[i + 1], contents[i + 2], ]
                            });
                        }
                    }
                }
            }

            if let Some(accessor) = primitive.indices() {
                let buffer_view = accessor.view().unwrap();
                let contents: &[u16] = bytemuck::cast_slice(&example.buffers[0][buffer_view.offset()..(buffer_view.offset() + buffer_view.length())]);
                for val in contents {
                    indices.push(*val);
                }
            }
        }
    }

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        memory_allocator.as_ref(),
        BufferUsage {
            vertex_buffer: true,
            ..Default::default()
        },
        false,
        vertices.into_iter(),
    )
        .expect("failed to create buffer");

    let index_buffer = CpuAccessibleBuffer::from_iter(
        memory_allocator.as_ref(),
        BufferUsage {
            index_buffer: true,
            ..Default::default()
        },
        false,
        indices.into_iter(),
    ).expect("failed to create index buffer");

    let window = app.surface.object().unwrap().downcast_ref::<Window>().unwrap();
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: window.inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let sphere_scale = 0.12f32;

    let mut camera = Camera::new(vec3(0.0, 0.0, -85.0), 0.01);

    let aspect_ratio =
        app.swapchain.image_extent()[0] as f32 / app.swapchain.image_extent()[1] as f32;
    let projection = perspective(aspect_ratio, f32::to_radians(std::f32::consts::FRAC_PI_2), 0.01, 256.0);
    let data = UBO {
        model: identity::<f32, 4>() * Mat4::from_diagonal(&vec4(sphere_scale, sphere_scale, sphere_scale, 1.0)),
        view: camera.get_view_matrix(),
        projection,
    };

    let ubo = CpuAccessibleBuffer::from_data(
        memory_allocator.as_ref(),
        BufferUsage {
            uniform_buffer: true,
            ..Default::default()
        },
        false,
        data,
    ).unwrap();

    let pipeline = get_pipeline(
        app.device.clone(),
        vs_shader.clone(),
        fs_shader.clone(),
        render_pass.clone(),
        viewport.clone(),
    );

    let layout = pipeline.layout().set_layouts().get(0).unwrap();

    let descriptor_allocator = StandardDescriptorSetAllocator::new(app.device.clone());
    let set = PersistentDescriptorSet::new(
        &descriptor_allocator,
        layout.clone(),
        [WriteDescriptorSet::buffer(0, ubo.clone())],
    ).unwrap();

    let mut command_buffers = get_command_buffers(&app, &example, &pipeline, &framebuffers, &vertex_buffer, &index_buffer, &set);

    let mut recreate_swapchain = true;

    let mut now_keys = [false; 255];
    let mut prev_keys = now_keys;

    let mut mouse_pressed: (MouseButton, bool) = (MouseButton::Left, false);

    let mut last_frame = Instant::now();

    event_loop.run(move |event, _, control_flow| match event {
        Event::NewEvents(_) => {
            prev_keys.copy_from_slice(&now_keys);
        }
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::ExitWithCode(0);
        }
        Event::WindowEvent {
            event: WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    virtual_keycode: Some(keycode),
                    state,
                    ..
                },
                ..
            },
            ..
        } => {
            match state {
                ElementState::Pressed => {
                    now_keys[keycode as usize] = true;
                }
                ElementState::Released => {
                    now_keys[keycode as usize] = false;
                }
            }
        }
        Event::WindowEvent {
            event: MouseWheel {
                delta: MouseScrollDelta::LineDelta(_x, y),
                ..
            },
            ..
        } => {
            camera.translate(vec3(0.0, 0.0, y * 2.0));
        }
        Event::DeviceEvent {
            event: MouseMotion {
                delta: (x, y)
            },
            ..
        } => {
            let scale = 1.0;
            if mouse_pressed.1 {
                camera.rotation.x -= y as f32 / scale;
                camera.rotation.y -= x as f32 / scale;
            }
        }
        Event::DeviceEvent {
            event: Button {
                state,
                ..
            },
            ..
        } => {
            mouse_pressed.1 = state == ElementState::Pressed;
        }
        Event::MainEventsCleared => {
            let mut write_buffer = ubo.write().unwrap();

            camera.update_view_matrix();
            write_buffer.view = camera.get_view_matrix();
        }
        Event::RedrawRequested(..) => {
            let elapsed = last_frame.elapsed().as_millis();
            if elapsed < (1000.0 / 60.0) as u128 {
                return;
            } else {
                last_frame = Instant::now();
            }


            let window = app.surface.object().unwrap().downcast_ref::<Window>().unwrap();

            if window.inner_size().width == 0 {
                return;
            }

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
            let new_framebuffers = app.get_framebuffers(&memory_allocator, &new_images, &render_pass);

            viewport.dimensions = window.inner_size().into();
            let new_pipeline = get_pipeline(
                app.device.clone(),
                vs_shader.clone(),
                fs_shader.clone(),
                render_pass.clone(),
                viewport.clone(),
            );
            command_buffers =
                get_command_buffers(&app, &example, &new_pipeline, &new_framebuffers, &vertex_buffer, &index_buffer, &set);


            let (image_i, _suboptimal, acquire_future) =
                match swapchain::acquire_next_image(app.swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {:?}", e),
                };


            let execution = sync::now(app.device.clone())
                .join(acquire_future)
                .then_execute(app.queue.clone(), command_buffers[image_i as usize].clone())
                .unwrap()
                .then_swapchain_present(
                    app.queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(app.swapchain.clone(), image_i),
                )
                .then_signal_fence_and_flush();

            match execution {
                Ok(future) => {
                    future.wait(None).unwrap();
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                }
                Err(e) => {
                    println!("failed to flush future: {:?}", e);
                }
            }
        }
        _ => {
            let window = app.surface.object().unwrap().downcast_ref::<Window>().unwrap();
            window.request_redraw();
        }
    });
}

fn get_command_buffers(
    app: &App,
    example: &Example,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    vertex_buffer: &Arc<CpuAccessibleBuffer<[Vertex]>>,
    index_buffer: &Arc<CpuAccessibleBuffer<[u16]>>,
    set: &Arc<PersistentDescriptorSet>,
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
                .bind_index_buffer(index_buffer.clone())
                .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline.layout().clone(), 0, set.clone());

            for push_constant in example.spheres {
                builder
                    .push_constants(pipeline.layout().clone(), 0, push_constant)
                    .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0).unwrap();
            }

            builder.end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}