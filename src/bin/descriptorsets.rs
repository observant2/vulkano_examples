use std::io::Cursor;
use std::process::exit;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use ktx::KtxInfo;
use nalgebra_glm::{identity, Mat4, perspective, rotate, scale, translate, vec3, Vec3, Vec4, vec4};
use vulkano_examples::{App};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassContents};
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
use vulkano::image::{ImageDimensions, ImmutableImage, MipmapsCount};
use vulkano::image::view::ImageView;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::sampler::{Sampler, SamplerCreateInfo};
use winit::event::{ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event::DeviceEvent::{Button, MouseMotion};
use winit::event::WindowEvent::MouseWheel;
use winit::event_loop::{ControlFlow};
use winit::window::{Window};
use vulkano_examples::camera::Camera;
use vulkano_examples::gltf_loader::Model;

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

vulkano::impl_vertex!(Vertex, position, uv);

struct Cube {
    matrices: Matrices,
    rotation: Vec3,
    descriptor_set: Option<Arc<PersistentDescriptorSet>>,
    buffer: Option<Arc<CpuAccessibleBuffer<Matrices>>>,
}

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct Matrices {
    projection: Mat4,
    view: Mat4,
    model: Mat4,
}

struct Example {
    model: Model,
    cubes: [Cube; 2],
}

impl Example {
    fn new(camera: &Camera) -> Self {
        let model = Model::load("./assets/models/cube.gltf");

        let cube1 = Cube {
            matrices: Matrices {
                model: identity(),
                view: camera.get_view_matrix(),
                projection: camera.get_perspective_matrix(),
            },
            rotation: Vec3::zeros(),
            descriptor_set: None,
            buffer: None,
        };

        let cube2 = Cube {
            matrices: Matrices {
                model: identity(),
                view: camera.get_view_matrix(),
                projection: camera.get_perspective_matrix(),
            },
            rotation: Vec3::zeros(),
            descriptor_set: None,
            buffer: None,
        };

        Example {
            model,
            cubes: [cube1, cube2],
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

layout (set = 0, binding = 0) uniform UBO
{
	mat4 projection;
	mat4 view;
	mat4 model;
} ubo;

layout (location = 0) out vec2 outUV;

void main()
{
	outUV = uv;
	gl_Position =  ubo.projection * ubo.view * ubo.model * vec4(position, 1.0);
}
"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

layout (set = 0, binding = 1) uniform sampler2D samplerColorMap;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

void main()
{
	outFragColor = vec4(vec3(2), 1) * texture(samplerColorMap, inUV);
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

pub fn main() {
    let (mut app, event_loop) = App::new("descriptorsets");

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
                format: Format::D32_SFLOAT,
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
    let projection = perspective(aspect_ratio, f32::to_radians(60.0), 0.01, 512.0);
    let mut camera = Camera::new(vec3(0.0, 0.0, -5.0), projection);
    let mut example = Example::new(&camera);

    let vertices = example.model.vertices.into_iter().zip(example.model.tex_coords).map(|(v, t)| Vertex {
        position: v,
        uv: t,
    });

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        memory_allocator.as_ref(),
        BufferUsage {
            vertex_buffer: true,
            ..Default::default()
        },
        false,
        vertices,
    )
        .expect("failed to create buffer");

    let index_buffer = CpuAccessibleBuffer::from_iter(
        memory_allocator.as_ref(),
        BufferUsage {
            index_buffer: true,
            ..Default::default()
        },
        false,
        example.model.indices,
    ).expect("failed to create index buffer");

    let window = app.surface.object().unwrap().downcast_ref::<Window>().unwrap();
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: window.inner_size().into(),
        depth_range: 0.0..1.0,
    };

    for cube in &mut example.cubes {
        cube.buffer = Some(CpuAccessibleBuffer::from_data(
            memory_allocator.as_ref(),
            BufferUsage {
                uniform_buffer: true,
                ..Default::default()
            },
            false,
            cube.matrices,
        ).unwrap());
    }


    let mut pipeline = get_pipeline(
        app.device.clone(),
        vs_shader.clone(),
        fs_shader.clone(),
        render_pass.clone(),
        viewport.clone(),
    );

    let layout = pipeline.layout().set_layouts().get(0).unwrap();

    let descriptor_allocator = StandardDescriptorSetAllocator::new(app.device.clone());

    let mut uploads = AutoCommandBufferBuilder::primary(
        &app.allocator_command_buffer,
        app.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
        .unwrap();

    let textures = vec![
        include_bytes!("../../assets/textures/crate01_color_height_rgba.ktx").to_vec(),
        include_bytes!("../../assets/textures/crate02_color_height_rgba.ktx").to_vec(),
    ]
        .into_iter()
        .map(|png_bytes| {
            let cursor = Cursor::new(png_bytes);
            let decoder = ktx::Decoder::new(cursor).unwrap();
            let mut image_data = Vec::new();
            let width = decoder.pixel_width();
            let height = decoder.pixel_height();
            image_data = decoder.read_textures().next().unwrap();
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
        })
        .collect::<Vec<Arc<ImmutableImage>>>();

    let sampler = Sampler::new(app.device.clone(), SamplerCreateInfo::simple_repeat_linear()).unwrap();

    for (i, cube) in example.cubes.iter_mut().enumerate() {
        cube.descriptor_set = Some(PersistentDescriptorSet::new(
            &descriptor_allocator,
            layout.clone(),
            [WriteDescriptorSet::buffer(0, cube.buffer.as_ref().unwrap().clone()),
                WriteDescriptorSet::image_view_sampler(1, ImageView::new_default(textures[i].clone()).unwrap(), sampler.clone())],
        ).unwrap());
    }

    let _ = uploads
        .build()
        .unwrap()
        .execute(app.queue.clone())
        .unwrap()
        .then_signal_fence_and_flush();

    let mut command_buffers = get_command_buffers(&app, &pipeline, &framebuffers, &vertex_buffer, &index_buffer, &example.cubes);

    let mut recreate_swapchain = true;

    let mut mouse_pressed: (MouseButton, bool) = (MouseButton::Left, false);

    let mut last_frame = Instant::now();

    let mut previous_frame_end = Some(sync::now(app.device.clone()).boxed());

    event_loop.run(move |event, _, control_flow| match event {
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
                camera.rotation.x += y as f32 / scale;
                camera.rotation.y += x as f32 / scale;
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
                example.cubes[0].rotation.x += 1.2;
                if example.cubes[0].rotation.x > 360.0 {
                    example.cubes[0].rotation.x -= 360.0;
                }
                example.cubes[1].rotation.y += 0.8;
                if example.cubes[1].rotation.y > 360.0 {
                    example.cubes[1].rotation.y -= 360.0;
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
                let perspective = perspective(aspect_ratio, f32::to_radians(60.0), 0.01, 512.0);
                camera.set_perspective(perspective);

                command_buffers = get_command_buffers(&app, &pipeline, &framebuffers, &vertex_buffer, &index_buffer, &example.cubes);

                recreate_swapchain = false;
            }

            {
                let cube1 = example.cubes[0].buffer.as_ref().unwrap().write();
                let cube2 = example.cubes[1].buffer.as_ref().unwrap().write();

                if cube1.is_ok() && cube2.is_ok() {
                    let mut cube1 = cube1.unwrap();
                    let mut cube2 = cube2.unwrap();
                    cube1.model = translate(&identity(), &vec3(-2.0, 0.0, 0.0));
                    cube2.model = translate(&identity(), &vec3(1.5, 0.5, 0.0));

                    for (i, cube) in [cube1, cube2].iter_mut().enumerate() {
                        cube.projection = camera.get_perspective_matrix();
                        cube.view = camera.get_view_matrix();
                        cube.model = rotate(&cube.model, f32::to_radians(example.cubes[i].rotation.x), &Vec3::x_axis());
                        cube.model = rotate(&cube.model, f32::to_radians(example.cubes[i].rotation.y), &Vec3::y_axis());
                        cube.model = rotate(&cube.model, f32::to_radians(example.cubes[i].rotation.z), &Vec3::z_axis());
                        cube.model = scale(&cube.model, &Vec3::from_element(0.25));
                    }
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
    });
}

fn get_command_buffers(
    app: &App,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    vertex_buffer: &Arc<CpuAccessibleBuffer<[Vertex]>>,
    index_buffer: &Arc<CpuAccessibleBuffer<[u16]>>,
    cubes: &[Cube; 2],
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
                            Some(ClearValue::Depth(1.0)),
                        ],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassContents::Inline,
                ).unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .bind_index_buffer(index_buffer.clone());
            for cube in cubes {
                builder
                    .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline.layout().clone(), 0,
                                          vec![cube.descriptor_set.as_ref().unwrap().clone(),
                                          ])
                    .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0).unwrap();
            }

            builder.end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}
