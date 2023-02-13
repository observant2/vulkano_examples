use std::sync::Arc;
use nalgebra_glm::identity;
use vulkano::buffer::{Buffer, BufferAllocateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::{AttachmentImage, ImageAccess, ImageUsage, SwapchainImage};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano_examples::{App, gltf_loader};
use crate::shaders::{fs_gbuffer, vs_gbuffer};
use crate::shaders::vs_gbuffer::ty::ModelViewProjection;

const DEPTH_FORMAT: Format = Format::D32_SFLOAT;
const POSITION_FORMAT: Format = Format::R16G16B16A16_SFLOAT;
const NORMAL_FORMAT: Format = Format::R8G8B8A8_UNORM;
const ALBEDO_FORMAT: Format = Format::R8G8B8A8_UNORM;

pub struct GbufferPass {
    pub view_projection_set: Vec<Arc<PersistentDescriptorSet>>,
    pub view_projection_buffer: Subbuffer<ModelViewProjection>,
    pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
}

impl GbufferPass {
    pub fn new(app: &App, viewport: &Viewport) -> Self {
        let view_projection_buffer = Buffer::from_data(
            &app.allocator_memory,
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

        let render_pass = GbufferPass::create_render_pass(&app);
        let pipeline = GbufferPass::create_graphics_pipeline(app.device.clone(), render_pass.clone(), viewport.clone());
        let framebuffers = GbufferPass::create_framebuffers(app.allocator_memory.clone(), &app.swapchain_images, render_pass.clone());
        let view_projection_set = vec![PersistentDescriptorSet::new(
            &app.allocator_descriptor_set,
            pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, view_projection_buffer.clone()),
            ],
        ).unwrap()];

        Self {
            pipeline,
            render_pass,
            framebuffers,
            view_projection_buffer,
            view_projection_set,
        }
    }

    fn create_graphics_pipeline(device: Arc<Device>, render_pass: Arc<RenderPass>, viewport: Viewport) -> Arc<GraphicsPipeline> {
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

    fn create_render_pass(app: &App) -> Arc<RenderPass> {
        vulkano::ordered_passes_renderpass!(
        app.device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: app.swapchain.image_format(),
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
            .unwrap()
    }

    fn create_framebuffers(memory_allocator: Arc<StandardMemoryAllocator>, images: &[Arc<SwapchainImage>], render_pass: Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
        images
            .iter()
            .map(|image| {
                let color = ImageView::new_default(AttachmentImage::with_usage(
                    &memory_allocator,
                    image.dimensions().width_height(), image.format(), ImageUsage::SAMPLED | image.usage()).unwrap()).unwrap();
                let position = ImageView::new_default(
                    AttachmentImage::with_usage(&memory_allocator,
                                                image.dimensions().width_height(), POSITION_FORMAT, ImageUsage::SAMPLED).unwrap()
                ).unwrap();
                let normal = ImageView::new_default(
                    AttachmentImage::with_usage(&memory_allocator,
                                                image.dimensions().width_height(), NORMAL_FORMAT, ImageUsage::SAMPLED).unwrap()
                ).unwrap();
                let albedo = ImageView::new_default(
                    AttachmentImage::with_usage(&memory_allocator,
                                                image.dimensions().width_height(), ALBEDO_FORMAT, ImageUsage::SAMPLED).unwrap()
                ).unwrap();
                let depth_buffer = ImageView::new_default(
                    AttachmentImage::transient(&memory_allocator,
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

    pub fn get_graphics_pipeline(&self) -> Arc<GraphicsPipeline> {
        self.pipeline.clone()
    }

    pub fn get_framebuffers(&self) -> Vec<Arc<Framebuffer>> {
        self.framebuffers.clone()
    }

    pub fn get_descriptor_sets(&self) -> Vec<Arc<PersistentDescriptorSet>> {
        self.view_projection_set.clone()
    }

    pub fn recreate_resources(&mut self, app: &App, viewport: &Viewport) {
        self.pipeline = GbufferPass::create_graphics_pipeline(app.device.clone(), self.render_pass.clone(), viewport.clone());
        self.framebuffers = GbufferPass::create_framebuffers(app.allocator_memory.clone(), &app.swapchain_images, self.render_pass.clone());
    }
}
