mod shaders;
mod gbuffer_pipeline;
mod ssao_pipeline;
mod composition_pipeline;

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
use vulkano::sampler::{Filter, Sampler, SamplerCreateInfo, SamplerMipmapMode};
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
use crate::composition_pipeline::CompositionPass;
use crate::gbuffer_pipeline::GbufferPass;

use crate::shaders::{fs_composition, vs_fullscreen};
use crate::shaders::fs_ssao::ty::Projection;
use crate::ssao_pipeline::SsaoPass;

struct Example {
    gbuffer_pass: GbufferPass,
    ssao_pass: SsaoPass,
    composition_pass: CompositionPass,

    scene: Scene,
}

pub fn main() {
    let (mut app, event_loop) = App::new("SSAO");

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(app.device.clone()));

    let aspect_ratio =
        app.swapchain.image_extent()[0] as f32 / app.swapchain.image_extent()[1] as f32;

    let scene = Scene::load("./data/models/treasure_smooth.gltf", &memory_allocator, true, true);

    let window = app.surface.object().unwrap().downcast_ref::<Window>().unwrap();
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: window.inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let gbuffer_pass = GbufferPass::new(&app, &viewport);
    let ssao_pass = SsaoPass::new(&app, &viewport, &gbuffer_pass);
    let composition_pass = CompositionPass::new(&app, &viewport, &gbuffer_pass, &ssao_pass);

    let mut example = Example {
        gbuffer_pass,
        ssao_pass,
        composition_pass,

        scene,
    };

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

    let mut command_buffers = get_command_buffers(&app, &example);

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

                    viewport.dimensions = window.inner_size().into();

                    // swapchain recreation invalidated framebuffers, descriptors..., recreate them:
                    example.gbuffer_pass.recreate_resources(&app, &viewport);
                    example.ssao_pass.recreate_resources(&app, &viewport, &example.gbuffer_pass);
                    example.composition_pass.recreate_resources(&app, &viewport, &example.gbuffer_pass, &example.ssao_pass);

                    command_buffers = get_command_buffers(&app, &example);

                    recreate_swapchain = false;
                }

                {
                    let view_projection = example.gbuffer_pass.view_projection_buffer.write();

                    if let Ok(mut view_projection) = view_projection {
                        view_projection.view = camera.get_view_matrix().data.0;
                        view_projection.projection = camera.get_perspective_matrix().data.0;
                    }

                    let projection = example.ssao_pass.projection_buffer.write();
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

fn get_command_buffers(
    app: &App,
    example: &Example,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    example.gbuffer_pass.get_framebuffers()
        .iter()
        .zip(&example.ssao_pass.get_framebuffers())
        .zip(&example.composition_pass.get_framebuffers())
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

            builder.bind_pipeline_graphics(example.gbuffer_pass.get_graphics_pipeline())
                .bind_descriptor_sets(PipelineBindPoint::Graphics,
                                      example.gbuffer_pass.get_graphics_pipeline().layout().clone(),
                                      0, vec![example.gbuffer_pass.view_projection_set[0].clone()]);
            example.scene.draw(&mut builder);
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
                .bind_pipeline_graphics(example.ssao_pass.get_graphics_pipeline())
                .bind_descriptor_sets(PipelineBindPoint::Graphics,
                                      example.ssao_pass.get_graphics_pipeline().layout().clone(),
                                      0, vec![example.ssao_pass.ssao_set.as_ref().unwrap()[i].clone()])
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
                .bind_pipeline_graphics(example.composition_pass.get_graphics_pipeline())
                .bind_descriptor_sets(PipelineBindPoint::Graphics,
                                      example.composition_pass.get_graphics_pipeline().layout().clone(),
                                      0, vec![example.composition_pass.composition_set.as_ref().unwrap()[i].clone()])
                .draw(3, 1, 0, 0).unwrap();

            builder
                .end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}
