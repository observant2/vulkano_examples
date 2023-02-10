use std::collections::btree_map::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Zeroable};
use nalgebra_glm::{identity, Mat4, normalize, vec3, Vec3, vec3_to_vec4, Vec4, vec4};
use rand::Rng;
use vulkano::{swapchain, sync};
use vulkano::buffer::{BufferUsage, Buffer, Subbuffer, BufferAllocateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyImageInfo, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorType};
use vulkano::device::Device;
use vulkano::format::{ClearValue, Format};
use vulkano::image::{AttachmentImage, ImageAccess, ImageUsage, ImmutableImage, SwapchainImage};
use vulkano::image::view::{ImageView, ImageViewCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode};
use vulkano::sampler::Filter::Nearest;
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

use crate::shaders::{fs_composition, fs_gbuffer, fs_ssao, vs_fullscreen, vs_gbuffer};
use crate::shaders::fs_composition::ty::SsaoSettings;
use crate::shaders::fs_gbuffer::ty::ModelViewProjection;
use crate::shaders::fs_ssao::ty::{Projection, SsaoKernel};

mod shaders;

const SSAO_NOISE_DIM: usize = 4;

struct Example {
    ssao_settings: Subbuffer<SsaoSettings>,
    ssao_kernel: Subbuffer<SsaoKernel>,
    ssao_noise: Arc<ImageView<ImmutableImage>>,

    composition_scene: Scene,
    view_projection_set: Arc<PersistentDescriptorSet>,

    gbuffer_framebuffers: Vec<Arc<Framebuffer>>,
    ssao_framebuffers: Vec<Arc<Framebuffer>>,
    composition_framebuffers: Vec<Arc<Framebuffer>>,

    sampler: Arc<Sampler>,
}

pub fn create_ssao_kernel() -> SsaoKernel {
    let lerp = |a: f32, b: f32, f: f32|
        {
            a + f * (b - a)
        };

    let mut rnd = rand::thread_rng();

    // Sample kernel
    let mut samples = [Vec4::identity().data.0[0]; 64];
    for i in 0..64
    {
        // Generate random point in hemisphere in z direction
        let mut sample = vec4(
            rnd.gen_range(-1.0..=1.0),
            rnd.gen_range(-1.0..=1.0),
            rnd.gen_range(0.0..=1.0),
            0.0);
        sample = normalize(&sample);
        sample *= rnd.gen_range(0.0..=1.0);
        let mut scale = i as f32 / 64.0;
        scale = lerp(0.1, 1.0, scale * scale);
        sample *= scale;
        sample.w = 0.0;
        samples[i] = sample.data.0[0];
    }

    SsaoKernel {
        samples,
    }
}

pub fn create_noise_texture(app: &App) -> Arc<ImmutableImage> {
    let mut rnd = rand::thread_rng();

    let mut ssao_noise = [[0.0, 0.0, 0.0, 0.0]; SSAO_NOISE_DIM * SSAO_NOISE_DIM];
    for i in 0..ssao_noise.len()
    {
        ssao_noise[i] = vec4(rnd.gen_range(0.0..=1.0) as f32, rnd.gen_range(0.0..=1.0) as f32, 0.0, 0.0).data.0[0];
    }

    app.load_texture(bytemuck::cast_slice(&ssao_noise), SSAO_NOISE_DIM as u32, SSAO_NOISE_DIM as u32, 1, Format::R32G32B32A32_SFLOAT)
}


const DEPTH_FORMAT: Format = Format::D32_SFLOAT;
const POSITION_FORMAT: Format = Format::R16G16B16A16_SFLOAT;
const NORMAL_FORMAT: Format = Format::R8G8B8A8_UNORM;
const ALBEDO_FORMAT: Format = Format::R8G8B8A8_UNORM;

fn get_gbuffer_framebuffers(memory_allocator: &Arc<StandardMemoryAllocator>, images: &[Arc<SwapchainImage>], render_pass: &Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let color = ImageView::new_default(AttachmentImage::with_usage(
                memory_allocator,
                image.dimensions().width_height(), image.format(), ImageUsage::SAMPLED | image.usage()).unwrap()).unwrap();
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

fn get_fresh_framebuffer(memory_allocator: &Arc<StandardMemoryAllocator>, images: &[Arc<SwapchainImage>], render_pass: &Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let color = ImageView::new_default(AttachmentImage::with_usage(
                memory_allocator,
                image.dimensions().width_height(), image.format(), ImageUsage::SAMPLED | image.usage()).unwrap()).unwrap();

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

fn get_final_framebuffer(images: &[Arc<SwapchainImage>], render_pass: &Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
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

fn get_pipeline_ssao(
    app: &App,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    let vs = vs_fullscreen::load(app.device.clone()).unwrap();
    let fs = fs_ssao::load(app.device.clone()).unwrap();
    let render_pass = get_simple_render_pass(app);
    GraphicsPipeline::start()
        .vertex_input_state(gltf_loader::GltfVertex::per_vertex())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::TriangleList))
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), fs_ssao::SpecializationConstants {
            SSAO_RADIUS: 0.3,
            SSAO_KERNEL_SIZE: 64,
        })
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        .build(app.device.clone())
        .unwrap()
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
            color: [color, position, normal, albedo],
            depth_stencil: {depth},
            input: []
        }]
    )
        .unwrap();

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
            model: identity().data.0,
            view: identity().data.0,
            projection: identity().data.0,
        },
    ).unwrap();

    let projection_buffer = Buffer::from_data(
        memory_allocator.as_ref(),
        BufferAllocateInfo {
            buffer_usage: BufferUsage::UNIFORM_BUFFER,
            ..BufferAllocateInfo::default()
        },
        Projection {
            projection: identity().data.0,
        },
    ).unwrap();

    let layout = pipeline_gbuffer.layout().set_layouts().get(0).unwrap();

    let noise_texture = create_noise_texture(&app);

    let mut example = Example {
        ssao_settings: Buffer::from_data(
            memory_allocator.as_ref(),
            BufferAllocateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..BufferAllocateInfo::default()
            },
            SsaoSettings {
                ssao: 1,
                ssao_blur: 0,
                ssao_only: 1,
            },
        ).unwrap(),
        ssao_kernel: Buffer::from_data(
            memory_allocator.as_ref(),
            BufferAllocateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..BufferAllocateInfo::default()
            },
            create_ssao_kernel(),
        ).unwrap(),
        ssao_noise: ImageView::new(noise_texture.clone(), ImageViewCreateInfo {
            usage: ImageUsage::SAMPLED,
            ..ImageViewCreateInfo::from_image(&noise_texture)
        }).unwrap(),
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

        gbuffer_framebuffers: get_gbuffer_framebuffers(&memory_allocator, &app.swapchain_images, &render_pass),
        ssao_framebuffers: get_fresh_framebuffer(&app.allocator_memory, &app.swapchain_images, &get_simple_render_pass(&app)),
        composition_framebuffers: get_final_framebuffer(&app.swapchain_images, &get_simple_render_pass(&app)),
    };

    // Create ssao pipeline
    let mut pipeline_ssao = get_pipeline_ssao(&app, viewport.clone());
    let layout_ssao = pipeline_ssao.layout().set_layouts().get(0).unwrap();
    let ssao_sets = create_ssao_sets(&app.device, layout_ssao, &app.allocator_descriptor_set, &example, &projection_buffer);

    // Create composition pipeline
    let mut pipeline_composition = get_pipeline_composition(&app, viewport.clone());

    let layout_composition = pipeline_composition.layout().set_layouts().get(0).unwrap();
    let composition_sets = create_composition_sets(layout_composition, &app.allocator_descriptor_set, &example);

    let mut recreate_swapchain = true;

    let mut last_frame = Instant::now();

    let mut camera = {
        let mut camera = Camera::new(vec3(1.65, 1.75, -6.15), aspect_ratio, f32::to_radians(60.0), 0.1, 64.0);
        camera.set_rotation(vec3(-12.75, 380.0, 0.0));
        camera.camera_type = CameraType::FirstPerson;
        camera.movement_speed = 5.0;
        camera.update_view_matrix();
        camera
    };

    let mut previous_frame_end = Some(sync::now(app.device.clone()).boxed());

    let mut command_buffers = get_command_buffers(&app, &pipeline_gbuffer, &pipeline_ssao, &pipeline_composition, &example, &ssao_sets, &composition_sets);

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

                    // swapchain recreation invalidated framebuffers, recreate them:
                    example.gbuffer_framebuffers = get_gbuffer_framebuffers(&memory_allocator, &app.swapchain_images, &render_pass);
                    example.ssao_framebuffers = get_fresh_framebuffer(&app.allocator_memory, &app.swapchain_images, &get_simple_render_pass(&app));
                    example.composition_framebuffers = get_final_framebuffer(&app.swapchain_images, &get_simple_render_pass(&app));

                    viewport.dimensions = window.inner_size().into();
                    pipeline_gbuffer = get_pipeline_gbuffer(
                        app.device.clone(),
                        render_pass.clone(),
                        viewport.clone(),
                    );

                    pipeline_ssao = get_pipeline_ssao(&app, viewport.clone());

                    let layout_ssao = pipeline_ssao.layout().set_layouts().get(0).unwrap();
                    let ssao_sets = create_ssao_sets(&app.device, layout_ssao, &app.allocator_descriptor_set, &example, &projection_buffer);

                    pipeline_composition = get_pipeline_composition(&app, viewport.clone());
                    let layout_composition = pipeline_composition.layout().set_layouts().get(0).unwrap();
                    let composition_sets = create_composition_sets(layout_composition, &app.allocator_descriptor_set, &example);

                    command_buffers = get_command_buffers(&app, &pipeline_gbuffer, &pipeline_ssao, &pipeline_composition, &example, &ssao_sets, &composition_sets);

                    recreate_swapchain = false;
                }

                {
                    let view_projection = view_projection_buffer.write();

                    if let Ok(mut view_projection) = view_projection {
                        view_projection.view = camera.get_view_matrix().data.0;
                        view_projection.projection = camera.get_perspective_matrix().data.0;
                    }

                    let projection = projection_buffer.write();
                    if let Ok(mut projection) = projection {
                        projection.projection = camera.get_perspective_matrix().data.0;
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

fn create_ssao_sets(device: &Arc<Device>, layout_ssao: &Arc<DescriptorSetLayout>, descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>, example: &Example, projection: &Subbuffer<Projection>) ->
Vec<Arc<PersistentDescriptorSet>> {
    let noise_sampler = Sampler::new(device.clone(), SamplerCreateInfo {
        min_filter: Nearest,
        mag_filter: Nearest,
        address_mode: [SamplerAddressMode::Repeat; 3],
        ..SamplerCreateInfo::default()
    }).unwrap();

    example.gbuffer_framebuffers.iter().enumerate().map(|(i, f): (usize, &Arc<Framebuffer>)| {
        PersistentDescriptorSet::new(
            descriptor_set_allocator,
            layout_ssao.clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, f.attachments()[1].clone(), example.sampler.clone()), // position
                WriteDescriptorSet::image_view_sampler(1, f.attachments()[2].clone(), example.sampler.clone()), // normal
                WriteDescriptorSet::image_view_sampler(2, example.ssao_noise.clone(), noise_sampler.clone()), // ssao noise
                WriteDescriptorSet::buffer(3, example.ssao_kernel.clone()),
                WriteDescriptorSet::buffer(4, projection.clone()),
            ]).unwrap()
    }).collect()
}

fn create_composition_sets(layout_composition: &Arc<DescriptorSetLayout>, descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>, example: &Example) ->
Vec<Arc<PersistentDescriptorSet>> {
    // unfortunately we need a separate descriptor set for every framebuffer, because the
    // resulting image can be different for each renderpass and each framebuffer.
    example.gbuffer_framebuffers.iter().enumerate().map(|(i, f): (usize, &Arc<Framebuffer>)| {
        PersistentDescriptorSet::new(
            descriptor_set_allocator,
            layout_composition.clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, f.attachments()[1].clone(), example.sampler.clone()), // position
                WriteDescriptorSet::image_view_sampler(1, f.attachments()[2].clone(), example.sampler.clone()), // normal
                WriteDescriptorSet::image_view_sampler(2, f.attachments()[3].clone(), example.sampler.clone()), // albedo
                WriteDescriptorSet::image_view_sampler(3, example.ssao_framebuffers[i].attachments()[0].clone(), example.sampler.clone()), // ssao
                WriteDescriptorSet::image_view_sampler(4, f.attachments()[3].clone(), example.sampler.clone()), // TODO: ssao blur
                WriteDescriptorSet::buffer(5, example.ssao_settings.clone()),
            ]).unwrap()
    }).collect()
}

fn get_command_buffers(
    app: &App,
    pipeline_gbuffer: &Arc<GraphicsPipeline>,
    pipeline_ssao: &Arc<GraphicsPipeline>,
    pipeline_composition: &Arc<GraphicsPipeline>,
    example: &Example,
    ssao_sets: &[Arc<PersistentDescriptorSet>],
    composition_sets: &[Arc<PersistentDescriptorSet>],
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    example.gbuffer_framebuffers
        .iter()
        .zip(&example.ssao_framebuffers)
        .zip(&example.composition_framebuffers)
        .enumerate()
        .map(|(i, ((gbuffer_framebuffer, ssao_framebuffer), composition_framebuffer)): (_, ((&Arc<Framebuffer>, &Arc<Framebuffer>), _))| {
            let mut builder = AutoCommandBufferBuilder::primary(
                app.allocator_command_buffer.as_ref(),
                app.queue_family_index,
                CommandBufferUsage::MultipleSubmit,
            ).unwrap();

            // gbuffer pass
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
                        ..RenderPassBeginInfo::framebuffer(gbuffer_framebuffer.clone())
                    },
                    SubpassContents::Inline,
                ).unwrap();

            builder.bind_pipeline_graphics(pipeline_gbuffer.clone())
                .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_gbuffer.layout().clone(), 0, vec![example.view_projection_set.clone()]);
            example.composition_scene.draw(&mut builder);
            builder.end_render_pass().unwrap();

            // ssao pass
            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![
                            Some([0.1, 0.1, 0.1, 1.0].into()),
                        ],
                        ..RenderPassBeginInfo::framebuffer(ssao_framebuffer.clone())
                    },
                    SubpassContents::Inline,
                ).unwrap();

            builder
                .bind_pipeline_graphics(pipeline_ssao.clone())
                .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_ssao.layout().clone(), 0, vec![ssao_sets[i].clone()])
                .draw(3, 1, 0, 0).unwrap();

            builder
                .end_render_pass()
                .unwrap();

            // composition pass
            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![
                            Some([0.1, 0.1, 0.5, 1.0].into()),
                        ],
                        ..RenderPassBeginInfo::framebuffer(composition_framebuffer.clone())
                    },
                    SubpassContents::Inline,
                ).unwrap();

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
