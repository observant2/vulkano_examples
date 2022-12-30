use std::collections::btree_map::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use egui_winit_vulkano::{egui, Gui};
use egui_winit_vulkano::egui::{Color32, WidgetText};
use nalgebra_glm::{identity, Mat4, Vec2, vec2, vec3};
use vulkano::{swapchain, sync};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType};
use vulkano::device::Device;
use vulkano::format::{ClearValue, Format};
use vulkano::image::{AttachmentImage, ImageAccess, ImageAspect, ImageAspects, ImageSubresourceRange, ImageUsage, SwapchainImage};
use vulkano::image::view::{ImageView, ImageViewCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
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

use vulkano_examples::{App, gltf_loader};
use vulkano_examples::camera::{Camera, CameraType};
use vulkano_examples::gltf_loader::Scene;

use crate::shaders::{fs_read, fs_write, vs_read, vs_write};

mod shaders;

#[derive(PartialEq)]
enum AttachmentChoice {
    BrightnessContrast,
    DepthBuffer,
}

struct Example {
    brightness: f32,
    contrast: f32,
    range: Vec2,
    current_attachment: AttachmentChoice,
    postprocessing_buffer: Arc<CpuAccessibleBuffer<UBO>>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[gltf_loader::Vertex]>>,
    index_buffer: Arc<CpuAccessibleBuffer<[u16]>>,
    scene: Scene
}

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct ViewProjection {
    view: Mat4,
    projection: Mat4,
}

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct UBO {
    brightness_contrast: Vec2,
    range: Vec2,
    attachment_index: i32,
}

const DEPTH_FORMAT: Format = Format::D32_SFLOAT;

fn get_framebuffers(memory_allocator: &Arc<StandardMemoryAllocator>, images: &[Arc<SwapchainImage>], render_pass: &Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let color = ImageView::new_default(image.clone()).unwrap();
            let color_write = ImageView::new_default(
                AttachmentImage::with_usage(memory_allocator,
                                            image.dimensions().width_height(), image.format(),
                                            ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT).unwrap()
            ).unwrap();

            let depth_image = AttachmentImage::with_usage(memory_allocator,
                                                          image.dimensions().width_height(), DEPTH_FORMAT,
                                                          ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT).unwrap();
            let depth_buffer = ImageView::new(depth_image.clone(),
                                              ImageViewCreateInfo {
                                                  subresource_range: ImageSubresourceRange {
                                                      aspects: ImageAspects::DEPTH,
                                                      ..image.subresource_range()
                                                  },
                                                  ..ImageViewCreateInfo::from_image(&depth_image)
                                              },
            ).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![color, color_write, depth_buffer],
                    ..Default::default()
                },
            )
                .unwrap()
        })
        .collect::<Vec<_>>()
}

fn get_pipeline_write(
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
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        .build(device).unwrap()
}

fn get_pipeline_read(
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
                        (0, DescriptorSetLayoutBinding {
                            stages: ShaderStages::FRAGMENT,
                            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::InputAttachment)
                        }),
                        (1, DescriptorSetLayoutBinding {
                            stages: ShaderStages::FRAGMENT,
                            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::InputAttachment)
                        }),
                        (2, DescriptorSetLayoutBinding {
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
            color_write: {
                load: Clear,
                store: DontCare,
                format: color_format,
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
        {// regular rendering
            color: [color_write],
            depth_stencil: {depth},
            input: []
        },
        {// post processing
            color: [color],
            depth_stencil: {},
            input: [color_write, depth]
        }]
    )
        .unwrap();

    let mut framebuffers = get_framebuffers(&memory_allocator, &app.swapchain_images, &render_pass);

    let aspect_ratio =
        app.swapchain.image_extent()[0] as f32 / app.swapchain.image_extent()[1] as f32;

    let scene = Scene::load("./data/models/treasure_smooth.gltf", true);

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        memory_allocator.as_ref(),
        BufferUsage::VERTEX_BUFFER,
        false,
        scene.vertices.clone(),
    )
        .expect("failed to create buffer");

    let index_buffer = CpuAccessibleBuffer::from_iter(
        memory_allocator.as_ref(),
        BufferUsage::INDEX_BUFFER,
        false,
        scene.indices.clone(),
    ).expect("failed to create index buffer");


    // Create pipeline write
    let window = app.surface.object().unwrap().downcast_ref::<Window>().unwrap();
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: window.inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let vs_shader = vs_write::load(app.device.clone()).unwrap();
    let fs_shader = fs_write::load(app.device.clone()).unwrap();
    let mut pipeline_write = get_pipeline_write(
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

    let layout = pipeline_write.layout().set_layouts().get(0).unwrap();

    let view_projection_set = PersistentDescriptorSet::new(
        &app.allocator_descriptor_set,
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, view_projection_buffer.clone()),
        ],
    ).unwrap();

    let mut example = Example {
        brightness: 0.5,
        contrast: 1.8,
        range: vec2(0.98, 1.0),
        current_attachment: AttachmentChoice::BrightnessContrast,
        postprocessing_buffer: CpuAccessibleBuffer::from_data(
            memory_allocator.as_ref(),
            BufferUsage::UNIFORM_BUFFER,
            false,
            UBO {
                brightness_contrast: vec2(0.5, 1.8),
                range: vec2(0.0, 1.0),
                attachment_index: 0,
            },
        ).unwrap(),
        vertex_buffer,
        index_buffer,
        scene,
    };

    // Create pipeline read
    let pipeline_read = get_pipeline_read(
        app.device.clone(),
        vs_read::load(app.device.clone()).unwrap(),
        fs_read::load(app.device.clone()).unwrap(),
        render_pass.clone(),
        viewport.clone(),
    );

    let layout_read = pipeline_read.layout().set_layouts().get(0).unwrap();
    let postprocessing_sets = create_post_processing_sets(&framebuffers, &layout_read, &app.allocator_descriptor_set, &example);

    let mut recreate_swapchain = true;

    let mut last_frame = Instant::now();

    let mut previous_frame_end = Some(sync::now(app.device.clone()).boxed());

    let mut camera = Camera::new(vec3(1.65, 1.75, -6.15), aspect_ratio, f32::to_radians(60.0), 0.1, 256.0);
    camera.set_rotation(vec3(-12.75, 380.0, 0.0));
    camera.camera_type = CameraType::FirstPerson;
    camera.update_view_matrix();

    let mut gui = {
        Gui::new(
            &event_loop,
            app.surface.clone(),
            Some(color_format),
            app.queue.clone(),
            true,
        )
    };

    let mut command_buffers = get_command_buffers(&app, &pipeline_write, &pipeline_read, &framebuffers, &example, &view_projection_set, &postprocessing_sets);

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
            Event::RedrawRequested(..) => {
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
                    if let Ok(mut ubo) = example.postprocessing_buffer.write() {
                        ubo.brightness_contrast = vec2(example.brightness, example.contrast);
                        ubo.range = vec2(example.range.x, example.range.y);
                        ubo.attachment_index = if example.current_attachment == AttachmentChoice::BrightnessContrast { 0 } else { 1 };
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
                    pipeline_write = get_pipeline_write(
                        app.device.clone(),
                        vs_shader.clone(),
                        fs_shader.clone(),
                        render_pass.clone(),
                        viewport.clone(),
                    );

                    let pipeline_read = get_pipeline_read(
                        app.device.clone(),
                        vs_read::load(app.device.clone()).unwrap(),
                        fs_read::load(app.device.clone()).unwrap(),
                        render_pass.clone(),
                        viewport.clone(),
                    );

                    let layout_read = pipeline_read.layout().set_layouts().get(0).unwrap();
                    let postprocessing_sets = create_post_processing_sets(&framebuffers, &layout_read, &app.allocator_descriptor_set, &example);

                    command_buffers = get_command_buffers(&app, &pipeline_write, &pipeline_read, &framebuffers, &example, &view_projection_set, &postprocessing_sets);

                    recreate_swapchain = false;
                }

                {
                    let view_projection = view_projection_buffer.write();

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
                        previous_frame_end = Some(sync::now(app.device.clone()).boxed());
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

fn create_post_processing_sets(framebuffers: &Vec<Arc<Framebuffer>>, layout_read: &Arc<DescriptorSetLayout>, descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>, example: &Example) ->
Vec<Arc<PersistentDescriptorSet>> {
    // unfortunately we need a separate descriptor set for every framebuffer, because the
    // resulting image can be different for each renderpass and each framebuffer.
    framebuffers.iter().map(|f: &Arc<Framebuffer>| {
        PersistentDescriptorSet::new(
            descriptor_set_allocator,
            layout_read.clone(),
            [
                WriteDescriptorSet::image_view(0, f.attachments()[1].clone()), // color
                WriteDescriptorSet::image_view(1, f.attachments()[2].clone()), // depth
                WriteDescriptorSet::buffer(2, example.postprocessing_buffer.clone()),
            ]).unwrap()
    }).collect()
}

fn get_command_buffers(
    app: &App,
    pipeline_write: &Arc<GraphicsPipeline>,
    pipeline_read: &Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    example: &Example,
    view_projection_set: &Arc<PersistentDescriptorSet>,
    postprocessing_sets: &[Arc<PersistentDescriptorSet>],
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .enumerate()
        .map(|(i, framebuffer)| {
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
                            Some([0.1, 0.1, 0.1, 1.0].into()),
                            Some(ClearValue::Depth(1.0)),
                        ],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassContents::Inline,
                ).unwrap();

            // Subpass 1
            builder.bind_pipeline_graphics(pipeline_write.clone())
                .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_write.layout().clone(), 0, vec![view_projection_set.clone()])
                .bind_vertex_buffers(0, example.vertex_buffer.clone())
                .bind_index_buffer(example.index_buffer.clone());
            example.scene.draw(&mut builder);

            // Subpass 2
            builder
                .next_subpass(SubpassContents::Inline).unwrap()
                .bind_pipeline_graphics(pipeline_read.clone())
                .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_read.layout().clone(), 0, vec![postprocessing_sets[i].clone()])
                .draw(3, 1, 0, 0).unwrap();

            builder
                .end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}

fn create_ui(gui: &mut Gui, example: &mut Example) {
    let ctx = gui.context();
    egui::Area::new("controls").movable(false).show(&ctx, |ui| {
        if ui.add(egui::widgets::Button::new(ui_text("Switch Attachment", Color32::GOLD))).clicked() {
            example.current_attachment = if example.current_attachment == AttachmentChoice::BrightnessContrast {
                AttachmentChoice::DepthBuffer
            } else {
                AttachmentChoice::BrightnessContrast
            };
        }

        match example.current_attachment {
            AttachmentChoice::BrightnessContrast => {
                ui.add(egui::widgets::Label::new(ui_text("Contrast and Brightness", Color32::WHITE)));
                ui.add(egui::widgets::Label::new(ui_text("brightness", Color32::WHITE)));
                ui.add(egui::Slider::new(&mut example.brightness, 0.0..=2.0));
                ui.add(egui::widgets::Label::new(ui_text("contrast", Color32::WHITE)));
                ui.add(egui::Slider::new(&mut example.contrast, 0.0..=4.0));
            }
            AttachmentChoice::DepthBuffer => {
                ui.add(egui::widgets::Label::new(ui_text("Depth", Color32::GREEN)));
                ui.add(egui::widgets::Label::new(ui_text("near", Color32::GREEN)));
                ui.add(egui::Slider::new(&mut example.range.x, 0.0..=1.0));
                ui.add(egui::widgets::Label::new(ui_text("far", Color32::GREEN)));
                ui.add(egui::Slider::new(&mut example.range.y, 0.0..=1.0));
            }
        }
    });
}

fn ui_text(str: &str, color: Color32) -> WidgetText {
    WidgetText::from(str).color(color)
}
