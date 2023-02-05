use std::collections::btree_map::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use egui_winit_vulkano::egui::Image;
use nalgebra_glm::{identity, Mat4, vec3, Vec3, vec3_to_vec4, Vec4, vec4};
use rand::Rng;
use vulkano::{swapchain, sync};
use vulkano::buffer::{BufferUsage, Buffer, Subbuffer, BufferAllocateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType};
use vulkano::descriptor_set::layout::DescriptorType::SampledImage;
use vulkano::device::Device;
use vulkano::format::{ClearValue, Format};
use vulkano::image::{AttachmentImage, ImageAccess, ImageUsage, SwapchainImage};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::sampler::{Filter, Sampler, SamplerCreateInfo, SamplerMipmapMode};
use vulkano::shader::{ShaderStages};
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

use crate::shaders::{fs_composition, fs_gbuffer, vs_fullscreen, vs_gbuffer};

mod shaders;

struct Example {
    ssao_settings: Subbuffer<SsaoSettings>,

    composition_scene: Scene,
    view_projection_set: Arc<PersistentDescriptorSet>,

    sampler: Arc<Sampler>,
}

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct ModelViewProjection {
    model: Mat4,
    view: Mat4,
    projection: Mat4,
}

#[repr(C)]
#[derive(Copy, Clone, Zeroable, Pod)]
struct SsaoSettings {
    ssao: i32,
    ssao_only: i32,
    ssao_blur: i32,
}

const DEPTH_FORMAT: Format = Format::D32_SFLOAT;
const POSITION_FORMAT: Format = Format::R16G16B16A16_SFLOAT;
const NORMAL_FORMAT: Format = Format::R16G16B16A16_SFLOAT;
const ALBEDO_FORMAT: Format = Format::R8G8B8A8_UNORM;

fn get_gbuffer_framebuffers(memory_allocator: &Arc<StandardMemoryAllocator>, images: &[Arc<SwapchainImage>], render_pass: &Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let color = ImageView::new_default(image.clone()).unwrap();
            let position = ImageView::new_default(
                AttachmentImage::with_usage(memory_allocator,
                                            image.dimensions().width_height(), POSITION_FORMAT, ImageUsage::SAMPLED).unwrap()
            ).unwrap();
            let normal = ImageView::new_default(
                AttachmentImage::with_usage(memory_allocator,
                                            image.dimensions().width_height(), NORMAL_FORMAT, ImageUsage::SAMPLED).unwrap()
            ).unwrap();
            let albedo = ImageView::new_default(
                AttachmentImage::with_usage(memory_allocator,
                                            image.dimensions().width_height(), ALBEDO_FORMAT, ImageUsage::SAMPLED).unwrap()
            ).unwrap();
            let depth_buffer = ImageView::new_default(
                AttachmentImage::transient(memory_allocator,
                                            image.dimensions().width_height(), DEPTH_FORMAT).unwrap()
            ).unwrap();

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

fn get_simple_framebuffers(images: &[Arc<SwapchainImage>], render_pass: &Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let color = ImageView::new_default(image.clone()).unwrap();

            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![color],
                    ..Default::default()
                },
            )
                .unwrap()
        })
        .collect::<Vec<_>>()
}

fn get_pipeline_gbuffer(
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    let vs = vs_gbuffer::load(device.clone()).unwrap();
    let fs = fs_gbuffer::load(device.clone()).unwrap();

    GraphicsPipeline::start()
        .vertex_input_state(gltf_loader::GltfVertex::per_vertex())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::TriangleList))
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        .build(device).unwrap()
}

fn get_pipeline_composition(
    app: &App,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    let vs = vs_fullscreen::load(app.device.clone()).unwrap();
    let fs = fs_composition::load(app.device.clone()).unwrap();
    let render_pass = get_simple_render_pass(app);
    GraphicsPipeline::start()
        .vertex_input_state(gltf_loader::GltfVertex::per_vertex())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        .build(app.device.clone())
        .unwrap()
}

fn get_simple_render_pass(app: &App) -> Arc<RenderPass> {
    vulkano::ordered_passes_renderpass!(
        app.device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: app.swapchain.image_format(),
                samples: 1,
            }
        },
        passes: [
        {
            color: [color],
            depth_stencil: {},
            input: []
        }]
    ).unwrap()
}

pub fn main() {
    let (mut app, event_loop) = App::new("SSAO");

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
                store: Store, // store for later use in subsequent render passes
                format: POSITION_FORMAT,
                samples: 1,
            },
            normal: {
                load: Clear,
                store: Store,
                format: NORMAL_FORMAT,
                samples: 1,
            },
            albedo: {
                load: Clear,
                store: Store,
                format: ALBEDO_FORMAT,
                samples: 1,
            },
            depth: {
                load: Clear,
                store: Store,
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
        }]
    )
        .unwrap();

    let mut framebuffers = get_gbuffer_framebuffers(&memory_allocator, &app.swapchain_images, &render_pass);

    let aspect_ratio =
        app.swapchain.image_extent()[0] as f32 / app.swapchain.image_extent()[1] as f32;

    let scene = Scene::load("./data/models/treasure_smooth.gltf", &memory_allocator, true, true);

    let window = app.surface.object().unwrap().downcast_ref::<Window>().unwrap();
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: window.inner_size().into(),
        depth_range: 0.0..1.0,
    };

    // Create gbuffer pipeline
    let mut pipeline_gbuffer = get_pipeline_gbuffer(
        app.device.clone(),
        render_pass.clone(),
        viewport.clone(),
    );

    let view_projection_buffer = Buffer::from_data(
        memory_allocator.as_ref(),
        BufferAllocateInfo {
            buffer_usage: BufferUsage::UNIFORM_BUFFER,
            ..BufferAllocateInfo::default()
        },
        ModelViewProjection {
            model: identity(),
            projection: identity(),
            view: identity(),
        },
    ).unwrap();

    let layout = pipeline_gbuffer.layout().set_layouts().get(0).unwrap();

    let mut example = Example {
        ssao_settings: Buffer::from_data(
            memory_allocator.as_ref(),
            BufferAllocateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..BufferAllocateInfo::default()
            },
            SsaoSettings {
                ssao: 0,
                ssao_blur: 1,
                ssao_only: 0,
            },
        ).unwrap(),
        view_projection_set: PersistentDescriptorSet::new(
            &app.allocator_descriptor_set,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, view_projection_buffer.clone()),
            ],
        ).unwrap(),
        composition_scene: scene,
        sampler: Sampler::new(app.device.clone(), SamplerCreateInfo {
            min_filter: Filter::Nearest,
            mag_filter: Filter::Nearest,
            mipmap_mode: SamplerMipmapMode::Nearest,
            ..SamplerCreateInfo::default()
        }).unwrap(),
    };

    // Create composition pipeline
    let pipeline_composition = get_pipeline_composition(&app, viewport.clone());
    let mut composition_framebuffers = get_simple_framebuffers(&app.swapchain_images, &get_simple_render_pass(&app));

    let layout_read = pipeline_composition.layout().set_layouts().get(0).unwrap();
    let composition_sets = create_composition_sets(&framebuffers, layout_read,
                                                   &app.allocator_descriptor_set, &example);

    let mut recreate_swapchain = true;

    let mut last_frame = Instant::now();

    let mut camera = {
        let mut camera = Camera::new(vec3(1.65, 1.75, -6.15), aspect_ratio, f32::to_radians(60.0), 0.1, 256.0);
        camera.set_rotation(vec3(-12.75, 380.0, 0.0));
        camera.camera_type = CameraType::FirstPerson;
        camera.movement_speed = 5.0;
        camera.update_view_matrix();
        camera
    };

    let mut previous_frame_end = Some(sync::now(app.device.clone()).boxed());

    let mut command_buffers = get_command_buffers(&app, &pipeline_gbuffer, &pipeline_composition, &framebuffers, &composition_framebuffers, &example, &composition_sets);

    event_loop.run(move |event, _, control_flow| {
        camera.handle_input(&event);

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

                    framebuffers = get_gbuffer_framebuffers(&memory_allocator, &app.swapchain_images, &render_pass);
                    composition_framebuffers = get_simple_framebuffers(&app.swapchain_images, &get_simple_render_pass(&app));

                    viewport.dimensions = window.inner_size().into();
                    pipeline_gbuffer = get_pipeline_gbuffer(
                        app.device.clone(),
                        render_pass.clone(),
                        viewport.clone(),
                    );

                    let pipeline_read = get_pipeline_composition(&app, viewport.clone());

                    let layout_read = pipeline_read.layout().set_layouts().get(0).unwrap();
                    let composition_sets = create_composition_sets(&framebuffers, layout_read, &app.allocator_descriptor_set, &example);

                    command_buffers = get_command_buffers(&app, &pipeline_gbuffer, &pipeline_read, &framebuffers, &composition_framebuffers, &example, &composition_sets);

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

fn create_composition_sets(framebuffers: &[Arc<Framebuffer>], layout_read: &Arc<DescriptorSetLayout>, descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>, example: &Example) ->
Vec<Arc<PersistentDescriptorSet>> {
    // unfortunately we need a separate descriptor set for every framebuffer, because the
    // resulting image can be different for each renderpass and each framebuffer.
    framebuffers.iter().map(|f: &Arc<Framebuffer>| {
        PersistentDescriptorSet::new(
            descriptor_set_allocator,
            layout_read.clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, f.attachments()[1].clone(), example.sampler.clone()), // position
                WriteDescriptorSet::image_view_sampler(1, f.attachments()[2].clone(), example.sampler.clone()), // normal
                WriteDescriptorSet::image_view_sampler(2, f.attachments()[3].clone(), example.sampler.clone()), // albedo
                WriteDescriptorSet::image_view_sampler(3, f.attachments()[3].clone(), example.sampler.clone()), // TODO: ssao
                WriteDescriptorSet::image_view_sampler(4, f.attachments()[3].clone(), example.sampler.clone()), // TODO: ssao blur
                WriteDescriptorSet::buffer(5, example.ssao_settings.clone()),
            ]).unwrap()
    }).collect()
}

fn get_command_buffers(
    app: &App,
    pipeline_gbuffer: &Arc<GraphicsPipeline>,
    pipeline_composition: &Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    composition_framebuffers: &[Arc<Framebuffer>],
    example: &Example,
    composition_sets: &[Arc<PersistentDescriptorSet>],
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .zip(composition_framebuffers)
        .enumerate()
        .map(|(i, (framebuffer, composition_framebuffer))| {
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
                .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_gbuffer.layout().clone(), 0, vec![example.view_projection_set.clone()]);
            example.composition_scene.draw(&mut builder);
            builder.end_render_pass().unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![
                            Some([0.1, 0.1, 0.1, 1.0].into()),
                        ],
                        ..RenderPassBeginInfo::framebuffer(composition_framebuffer.clone())
                    },
                    SubpassContents::Inline,
                ).unwrap();

            // Compose final image
            builder
                .bind_pipeline_graphics(pipeline_composition.clone())
                .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_composition.layout().clone(), 0, vec![composition_sets[i].clone()])
                .draw(3, 1, 0, 0).unwrap();

            builder
                .end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}
