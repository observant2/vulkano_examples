use std::collections::btree_map::BTreeMap;
use std::io::Cursor;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use egui_winit_vulkano::{Gui};
use egui_winit_vulkano::egui::{Color32, WidgetText};
use ktx::KtxInfo;
use nalgebra_glm::{identity, Mat4, vec3, Vec3, vec3_to_vec4, Vec4, vec4};
use rand::{Rng};
use vulkano::{swapchain, sync};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassContents};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType};
use vulkano::device::Device;
use vulkano::format::{ClearValue, Format};
use vulkano::image::{AttachmentImage, ImageAccess, ImageDimensions, ImageUsage, ImmutableImage, MipmapsCount, SwapchainImage};
use vulkano::image::view::{ImageView};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, StateMode};
use vulkano::pipeline::graphics::color_blend::ColorBlendState;
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthBoundsState, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::sampler::{Sampler, SamplerCreateInfo};
use vulkano::shader::{ShaderModule, ShaderStages};
use vulkano::swapchain::{
    AcquireError, SwapchainCreateInfo, SwapchainCreationError, SwapchainPresentInfo,
};
use vulkano::sync::{FlushError, GpuFuture};
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::window::Window;

use vulkano_examples::{App, gltf_loader};
use vulkano_examples::camera::{Camera, CameraType};
use vulkano_examples::gltf_loader::Scene;

use crate::shaders::{fs_composition, fs_gbuffer, fs_transparent, vs_composition, vs_gbuffer, vs_transparent};

mod shaders;

struct Example {
    composition_ubo: Arc<CpuAccessibleBuffer<UBO>>,
    composition_vertex_buffer: Arc<CpuAccessibleBuffer<[gltf_loader::Vertex]>>,
    composition_index_buffer: Arc<CpuAccessibleBuffer<[u16]>>,

    transparency_ubo: Arc<CpuAccessibleBuffer<ModelViewProjection>>,
    transparency_vertex_buffer: Arc<CpuAccessibleBuffer<[gltf_loader::Vertex]>>,
    transparency_index_buffer: Arc<CpuAccessibleBuffer<[u16]>>,

    composition_scene: Scene,
    transparency_scene: Scene,
}

impl Example {
    pub fn init_lights(&mut self) {
        let colors = [
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0),
            vec3(1.0, 1.0, 0.0),
        ];

        let mut rand = rand::thread_rng();

        if let Ok(mut ubo) = self.composition_ubo.write() {
            for light in &mut ubo.lights {
                light.position = vec4(
                    rand.gen_range(-6.0..6.0),
                    0.25 + rand.gen_range(0.0..4.0),
                    rand.gen_range(-6.0..6.0),
                    1.0,
                );
                light.color = colors[rand.gen_range(0..colors.len())];
                light.radius = rand.gen_range(1.0..2.0);
            }
        }
    }
}

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct ModelViewProjection {
    model: Mat4,
    view: Mat4,
    projection: Mat4,
}

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct Light {
    position: Vec4,
    color: Vec3,
    radius: f32,
}

const NUM_LIGHTS: usize = 64;

#[repr(C)]
#[derive(Copy, Clone, Zeroable, Pod)]
struct UBO {
    view_pos: Vec4,
    lights: [Light; NUM_LIGHTS],
}

impl Default for UBO {
    fn default() -> Self {
        UBO {
            view_pos: Vec4::zeros(),
            lights: [Light::default(); NUM_LIGHTS],
        }
    }
}

const DEPTH_FORMAT: Format = Format::D32_SFLOAT;
const POSITION_FORMAT: Format = Format::R16G16B16A16_SFLOAT;
const NORMAL_FORMAT: Format = Format::R16G16B16A16_SFLOAT;
const ALBEDO_FORMAT: Format = Format::R8G8B8A8_UNORM;

fn get_framebuffers(memory_allocator: &Arc<StandardMemoryAllocator>, images: &[Arc<SwapchainImage>], render_pass: &Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let color = ImageView::new_default(image.clone()).unwrap();
            let position = ImageView::new_default(
                AttachmentImage::with_usage(memory_allocator,
                                            image.dimensions().width_height(), POSITION_FORMAT,
                                            ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT).unwrap()
            ).unwrap();
            let normal = ImageView::new_default(
                AttachmentImage::with_usage(memory_allocator,
                                            image.dimensions().width_height(), NORMAL_FORMAT,
                                            ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT).unwrap()
            ).unwrap();
            let albedo = ImageView::new_default(
                AttachmentImage::with_usage(memory_allocator,
                                            image.dimensions().width_height(), ALBEDO_FORMAT,
                                            ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT).unwrap()
            ).unwrap();

            let depth_buffer = ImageView::new_default(AttachmentImage::transient(memory_allocator,
                                                                                 image.dimensions().width_height(), DEPTH_FORMAT).unwrap()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![color, position, normal, albedo, depth_buffer],
                    ..Default::default()
                },
            )
                .unwrap()
        })
        .collect::<Vec<_>>()
}

fn get_pipeline_gbuffer(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<gltf_loader::Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::TriangleList))
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        .build(device).unwrap()
}

fn get_pipeline_composition(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<gltf_loader::Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_pass, 1).unwrap())
        .with_pipeline_layout(device.clone(), PipelineLayout::new(device.clone(), PipelineLayoutCreateInfo {
            set_layouts: vec![
                DescriptorSetLayout::new(device.clone(), DescriptorSetLayoutCreateInfo {
                    bindings: BTreeMap::from([
                        // position
                        (0, DescriptorSetLayoutBinding {
                            stages: ShaderStages::FRAGMENT,
                            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::InputAttachment)
                        }),
                        // normal
                        (1, DescriptorSetLayoutBinding {
                            stages: ShaderStages::FRAGMENT,
                            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::InputAttachment)
                        }),
                        // albedo
                        (2, DescriptorSetLayoutBinding {
                            stages: ShaderStages::FRAGMENT,
                            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::InputAttachment)
                        }),
                        (3, DescriptorSetLayoutBinding {
                            stages: ShaderStages::FRAGMENT,
                            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
                        }),
                    ]),
                    ..DescriptorSetLayoutCreateInfo::default()
                }).unwrap()
            ],
            ..PipelineLayoutCreateInfo::default()
        }).unwrap())
        .unwrap()
}

fn get_pipeline_transparency(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<gltf_loader::Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .color_blend_state(ColorBlendState::default().blend_alpha())
        .render_pass(Subpass::from(render_pass, 2).unwrap())
        .with_pipeline_layout(device.clone(), PipelineLayout::new(device.clone(), PipelineLayoutCreateInfo {
            set_layouts: vec![
                DescriptorSetLayout::new(device.clone(), DescriptorSetLayoutCreateInfo {
                    bindings: BTreeMap::from([
                        (0, DescriptorSetLayoutBinding {
                            stages: ShaderStages::VERTEX,
                            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
                        }),
                        // position from gbuffer pass
                        (1, DescriptorSetLayoutBinding {
                            stages: ShaderStages::FRAGMENT,
                            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::InputAttachment)
                        }),
                        (2, DescriptorSetLayoutBinding {
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
    let (mut app, event_loop) = App::new("subpasses");

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
            position: {
                load: Clear,
                store: DontCare,
                format: POSITION_FORMAT,
                samples: 1,
            },
            normal: {
                load: Clear,
                store: DontCare,
                format: NORMAL_FORMAT,
                samples: 1,
            },
            albedo: {
                load: Clear,
                store: DontCare,
                format: ALBEDO_FORMAT,
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: DEPTH_FORMAT,
                samples: 1,
            }
        },
        passes: [
        {// gbuffer
            // TODO: is rendering to color actually necessary here?
            color: [color, position, normal, albedo],
            depth_stencil: {depth},
            input: []
        },
        {// composition
            color: [color],
            depth_stencil: {},
            input: [position, normal, albedo]
        },
        {// transparency
            color: [color],
            depth_stencil: {depth},
            input: [position] // contains depth in its alpha component
        }]
    )
        .unwrap();

    let mut framebuffers = get_framebuffers(&memory_allocator, &app.swapchain_images, &render_pass);

    let aspect_ratio =
        app.swapchain.image_extent()[0] as f32 / app.swapchain.image_extent()[1] as f32;

    let scene = Scene::load("./data/models/samplebuilding.gltf", true);

    let glass = Scene::load("./data/models/samplebuilding_glass.gltf", true);

    let mut uploads = AutoCommandBufferBuilder::primary(
        &app.allocator_command_buffer,
        app.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
        .unwrap();
    let glass_texture = {
        let bytes = include_bytes!("../../../data/textures/colored_glass_rgba.ktx").to_vec();
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

    let window = app.surface.object().unwrap().downcast_ref::<Window>().unwrap();
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: window.inner_size().into(),
        depth_range: 0.0..1.0,
    };
    let vs_shader = vs_transparent::load(app.device.clone()).unwrap();
    let fs_shader = fs_transparent::load(app.device.clone()).unwrap();
    let mut pipeline_transparency = get_pipeline_transparency(
        app.device.clone(),
        vs_shader.clone(),
        fs_shader.clone(),
        render_pass.clone(),
        viewport.clone(),
    );

    let transparency_layout = pipeline_transparency.layout().set_layouts().get(0).unwrap();
    let transparency_ubo = CpuAccessibleBuffer::from_data(
        memory_allocator.as_ref(),
        BufferUsage::UNIFORM_BUFFER,
        false,
        ModelViewProjection {
            model: identity(),
            projection: identity(),
            view: identity(),
        },
    ).unwrap();

    let _ = uploads
        .build()
        .unwrap()
        .execute(app.queue.clone())
        .unwrap()
        .then_signal_fence_and_flush();

    // Create gbuffer pipeline
    let vs_shader = vs_gbuffer::load(app.device.clone()).unwrap();
    let fs_shader = fs_gbuffer::load(app.device.clone()).unwrap();
    let mut pipeline_gbuffer = get_pipeline_gbuffer(
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
        ModelViewProjection {
            model: identity(),
            projection: identity(),
            view: identity(),
        },
    ).unwrap();

    let layout = pipeline_gbuffer.layout().set_layouts().get(0).unwrap();

    let view_projection_set = PersistentDescriptorSet::new(
        &app.allocator_descriptor_set,
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, view_projection_buffer.clone()),
        ],
    ).unwrap();

    let mut example = Example {
        composition_ubo: CpuAccessibleBuffer::from_data(
            memory_allocator.as_ref(),
            BufferUsage::UNIFORM_BUFFER,
            false,
            UBO::default(),
        ).unwrap(),
        composition_vertex_buffer: CpuAccessibleBuffer::from_iter(
        memory_allocator.as_ref(),
        BufferUsage::VERTEX_BUFFER,
        false,
        scene.vertices.clone(),
    ) .expect("failed to create buffer"),
        composition_index_buffer: CpuAccessibleBuffer::from_iter(
            memory_allocator.as_ref(),
            BufferUsage::INDEX_BUFFER,
            false,
            scene.indices.clone(),
        ).expect("failed to create index buffer"),
        composition_scene: scene,

        transparency_ubo: transparency_ubo.clone(),
        transparency_vertex_buffer: CpuAccessibleBuffer::from_iter(
            memory_allocator.as_ref(),
            BufferUsage::VERTEX_BUFFER,
            false,
            glass.vertices.clone(),
        ) .expect("failed to create buffer"),
        transparency_index_buffer:CpuAccessibleBuffer::from_iter(
            memory_allocator.as_ref(),
            BufferUsage::INDEX_BUFFER,
            false,
            glass.indices.clone(),
        ).expect("failed to create index buffer"),
        transparency_scene: glass,
    };

    let transparency_sets =
        create_transparency_sets(&framebuffers, &transparency_layout, &app.allocator_descriptor_set, &example, &glass_texture, &sampler);

    // Create composition pipeline
    let pipeline_composition = get_pipeline_composition(
        app.device.clone(),
        vs_composition::load(app.device.clone()).unwrap(),
        fs_composition::load(app.device.clone()).unwrap(),
        render_pass.clone(),
        viewport.clone(),
    );

    let layout_read = pipeline_composition.layout().set_layouts().get(0).unwrap();
    let composition_sets = create_composition_sets(&framebuffers, &layout_read,
                                                   &app.allocator_descriptor_set, &example);


    let mut recreate_swapchain = true;

    let mut last_frame = Instant::now();

    let mut previous_frame_end = Some(sync::now(app.device.clone()).boxed());

    let mut camera = {
        let mut camera = Camera::new(vec3(1.65, 1.75, -6.15), aspect_ratio, f32::to_radians(60.0), 0.1, 256.0);
        camera.set_rotation(vec3(-12.75, 380.0, 0.0));
        camera.camera_type = CameraType::FirstPerson;
        camera.movement_speed = 5.0;
        camera.update_view_matrix();
        camera
    };

    let mut gui = {
        Gui::new(
            &event_loop,
            app.surface.clone(),
            Some(color_format),
            app.queue.clone(),
            true,
        )
    };

    example.init_lights();

    let mut command_buffers = get_command_buffers(&app, &pipeline_gbuffer, &pipeline_composition, &pipeline_transparency, &framebuffers, &example, &view_projection_set, &composition_sets, &transparency_sets);

    event_loop.run(move |event, _, control_flow| {
        if let Event::WindowEvent {
            event: e,
            ..
        } = &event {
            if !gui.update(e) {
                camera.handle_input(&event);
            }
        } else {
            camera.handle_input(&event);
        }

        match &event {
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
            Event::RedrawEventsCleared => {
                let elapsed = last_frame.elapsed().as_millis();
                if elapsed < (1000.0 / 60.0) as u128 {
                    return;
                } else {
                    camera.update(elapsed as f32);
                    last_frame = Instant::now();
                }

                previous_frame_end.as_mut().unwrap().cleanup_finished();

                gui.immediate_ui(|gui: &mut Gui| {
                    create_ui(gui, &mut example);
                });

                {
                    if let Ok(mut ubo) = example.composition_ubo.write() {
                        let mut pos = vec3_to_vec4(&camera.get_position());
                        pos.w = 1.0;
                        ubo.view_pos = pos;
                    }
                }

                let window = app.surface.object().unwrap().downcast_ref::<Window>().unwrap();

                if window.inner_size().width == 0 || window.inner_size().height == 0 {
                    return;
                }

                if recreate_swapchain {
                    // TODO: Window resizing leads to memory leak!
                    let (new_swapchain, new_images) =
                        match app.swapchain.recreate(SwapchainCreateInfo {
                            image_extent: window.inner_size().into(),
                            ..app.swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("failed to recreate swapchain: {e:?}"),
                        };

                    app.swapchain = new_swapchain;
                    app.swapchain_images = new_images;

                    let aspect_ratio =
                        app.swapchain.image_extent()[0] as f32 / app.swapchain.image_extent()[1] as f32;
                    camera.set_perspective(aspect_ratio, f32::to_radians(60.0), 0.01, 512.0);

                    framebuffers = get_framebuffers(&memory_allocator, &app.swapchain_images, &render_pass);

                    viewport.dimensions = window.inner_size().into();
                    pipeline_gbuffer = get_pipeline_gbuffer(
                        app.device.clone(),
                        vs_shader.clone(),
                        fs_shader.clone(),
                        render_pass.clone(),
                        viewport.clone(),
                    );

                    let pipeline_read = get_pipeline_composition(
                        app.device.clone(),
                        vs_composition::load(app.device.clone()).unwrap(),
                        fs_composition::load(app.device.clone()).unwrap(),
                        render_pass.clone(),
                        viewport.clone(),
                    );

                    let layout_read = pipeline_read.layout().set_layouts().get(0).unwrap();
                    let postprocessing_sets = create_composition_sets(&framebuffers, &layout_read, &app.allocator_descriptor_set, &example);

                    command_buffers = get_command_buffers(&app, &pipeline_gbuffer, &pipeline_read, &pipeline_transparency, &framebuffers, &example,
                                                          &view_projection_set, &postprocessing_sets, &transparency_sets);

                    recreate_swapchain = false;
                }

                {
                    let view_projection = view_projection_buffer.write();

                    if let Ok(mut view_projection) = view_projection {
                        view_projection.view = camera.get_view_matrix();
                        view_projection.projection = camera.get_perspective_matrix();
                    }
                }
                {
                    let view_projection = transparency_ubo.write();

                    if let Ok(mut view_projection) = view_projection {
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
                        Err(e) => panic!("failed to acquire next image: {e:?}"),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                // execute command buffers
                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(app.queue.clone(), command_buffers[image_i as usize].clone())
                    .unwrap();

                // draw UI
                // TODO: There is a memory leak somewhere on resize
                let future = gui.draw_on_image(future, ImageView::new_default(app.swapchain_images[image_i as usize].clone()).unwrap());

                // present
                let future = future.then_swapchain_present(
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
                        let t = sync::now(app.device.clone());
                        previous_frame_end = Some(t.boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {e:?}");
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

fn create_composition_sets(framebuffers: &Vec<Arc<Framebuffer>>, layout_read: &Arc<DescriptorSetLayout>, descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>, example: &Example) ->
Vec<Arc<PersistentDescriptorSet>> {
    // unfortunately we need a separate descriptor set for every framebuffer, because the
    // resulting image can be different for each renderpass and each framebuffer.
    framebuffers.iter().map(|f: &Arc<Framebuffer>| {
        PersistentDescriptorSet::new(
            descriptor_set_allocator,
            layout_read.clone(),
            [
                WriteDescriptorSet::image_view(0, f.attachments()[1].clone()), // position
                WriteDescriptorSet::image_view(1, f.attachments()[2].clone()), // normal
                WriteDescriptorSet::image_view(2, f.attachments()[3].clone()), // albedo
                WriteDescriptorSet::buffer(3, example.composition_ubo.clone()),
            ]).unwrap()
    }).collect()
}

fn create_transparency_sets(framebuffers: &Vec<Arc<Framebuffer>>, layout_transparency: &Arc<DescriptorSetLayout>,
                            descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>, example: &Example,
                            glass_texture: &Arc<ImmutableImage>, sampler: &Arc<Sampler>) ->
Vec<Arc<PersistentDescriptorSet>> {
    framebuffers.iter().map(|f: &Arc<Framebuffer>| {
        PersistentDescriptorSet::new(
            descriptor_set_allocator,
            layout_transparency.clone(),
            [
                WriteDescriptorSet::buffer(0, example.transparency_ubo.clone()),
                WriteDescriptorSet::image_view(1, f.attachments()[1].clone()),
                WriteDescriptorSet::image_view_sampler(2, ImageView::new_default(glass_texture.clone()).unwrap(), sampler.clone())
            ]
        ).unwrap()
    }).collect()
}

fn get_command_buffers(
    app: &App,
    pipeline_gbuffer: &Arc<GraphicsPipeline>,
    pipeline_composition: &Arc<GraphicsPipeline>,
    pipeline_transparency: &Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    example: &Example,
    mvp_set: &Arc<PersistentDescriptorSet>,
    composition_sets: &[Arc<PersistentDescriptorSet>],
    transparency_sets: &[Arc<PersistentDescriptorSet>],
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .enumerate()
        .map(|(i, framebuffer)| {
            let mut builder = AutoCommandBufferBuilder::primary(
                app.allocator_command_buffer.as_ref(),
                app.queue_family_index,
                CommandBufferUsage::MultipleSubmit,
            ).unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![
                            Some([0.1, 0.1, 0.1, 1.0].into()),
                            Some([0.1, 0.1, 0.1, 1.0].into()),
                            Some([0.1, 0.1, 0.1, 1.0].into()),
                            Some([0.1, 0.1, 0.1, 1.0].into()),
                            Some(ClearValue::Depth(1.0)),
                        ],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassContents::Inline,
                ).unwrap();

            // Draw to gbuffer
            builder.bind_pipeline_graphics(pipeline_gbuffer.clone())
                .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_gbuffer.layout().clone(), 0, vec![mvp_set.clone()])
                .bind_vertex_buffers(0, example.composition_vertex_buffer.clone())
                .bind_index_buffer(example.composition_index_buffer.clone());
            example.composition_scene.draw(&mut builder);

            // Compose final image
            builder
                .next_subpass(SubpassContents::Inline).unwrap()
                .bind_pipeline_graphics(pipeline_composition.clone())
                .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_composition.layout().clone(), 0, vec![composition_sets[i].clone()])
                .draw(3, 1, 0, 0).unwrap();

            builder
                .next_subpass(SubpassContents::Inline).unwrap()
                .bind_pipeline_graphics(pipeline_transparency.clone())
                .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_transparency.layout().clone(), 0, vec![transparency_sets[i].clone()])
                .bind_vertex_buffers(0, example.transparency_vertex_buffer.clone())
                .bind_index_buffer(example.transparency_index_buffer.clone());
            example.transparency_scene.draw(&mut builder);

            builder
                .end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}

fn create_ui(_gui: &mut Gui, _example: &mut Example) {
    // let ctx = gui.context();
    // egui::Area::new("controls").movable(false).show(&ctx, |ui| {
    //     if ui.add(egui::widgets::Button::new(ui_text("Switch Attachment", Color32::GOLD))).clicked() {
    //         example.current_attachment = if example.current_attachment == AttachmentChoice::BrightnessContrast {
    //             AttachmentChoice::DepthBuffer
    //         } else {
    //             AttachmentChoice::BrightnessContrast
    //         };
    //     }
    //
    //     match example.current_attachment {
    //         AttachmentChoice::BrightnessContrast => {
    //             ui.add(egui::widgets::Label::new(ui_text("Contrast and Brightness", Color32::WHITE)));
    //             ui.add(egui::widgets::Label::new(ui_text("brightness", Color32::WHITE)));
    //             ui.add(egui::Slider::new(&mut example.brightness, 0.0..=2.0));
    //             ui.add(egui::widgets::Label::new(ui_text("contrast", Color32::WHITE)));
    //             ui.add(egui::Slider::new(&mut example.contrast, 0.0..=4.0));
    //         }
    //         AttachmentChoice::DepthBuffer => {
    //             ui.add(egui::widgets::Label::new(ui_text("Depth", Color32::GREEN)));
    //             ui.add(egui::widgets::Label::new(ui_text("near", Color32::GREEN)));
    //             ui.add(egui::Slider::new(&mut example.range.x, 0.0..=1.0));
    //             ui.add(egui::widgets::Label::new(ui_text("far", Color32::GREEN)));
    //             ui.add(egui::Slider::new(&mut example.range.y, 0.0..=1.0));
    //         }
    //     }
    // });
}

fn ui_text(str: &str, color: Color32) -> WidgetText {
    WidgetText::from(str).color(color)
}
