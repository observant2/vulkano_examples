use std::sync::Arc;

use nalgebra_glm::{Vec4, vec4};
use vulkano::buffer::{Buffer, BufferAllocateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::image::{ImageAccess, ImageUsage, ImmutableImage, SwapchainImage};
use vulkano::image::view::{ImageView, ImageViewCreateInfo};
use vulkano::pipeline::{GraphicsPipeline, Pipeline};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::sampler::{Filter, Sampler, SamplerCreateInfo};

use vulkano_examples::{App, gltf_loader};

use crate::gbuffer_pipeline::GbufferPass;
use crate::shaders::{fs_composition, vs_fullscreen};
use crate::shaders::fs_composition::ty::SsaoSettings;
use crate::ssao_pipeline::SsaoPass;

pub struct CompositionPass {
    pub composition_set: Option<Vec<Arc<PersistentDescriptorSet>>>,
    pub ssao_settings: Subbuffer<SsaoSettings>,

    pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
}

impl CompositionPass {
    pub fn new(app: &App, viewport: &Viewport, gbuffer_pass: &GbufferPass, ssao_pass: &SsaoPass) -> Self {
        let ssao_settings = Buffer::from_data(
            &app.allocator_memory,
            BufferAllocateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..BufferAllocateInfo::default()
            },
            SsaoSettings {
                ssao: 1,
                ssao_blur: 0,
                ssao_only: 0,
            },
        ).unwrap();

        let render_pass = CompositionPass::create_render_pass(&app);
        let pipeline = CompositionPass::create_graphics_pipeline(app.device.clone(), render_pass.clone(), viewport.clone());
        let framebuffers = CompositionPass::create_framebuffers(&app.swapchain_images, render_pass.clone());

        let mut ret = Self {
            pipeline,
            render_pass,
            framebuffers,
            ssao_settings,
            composition_set: None,
        };

        ret.create_composition_sets(app, gbuffer_pass, ssao_pass);

        ret
    }

    fn create_graphics_pipeline(device: Arc<Device>, render_pass: Arc<RenderPass>, viewport: Viewport) -> Arc<GraphicsPipeline> {
        let vs = vs_fullscreen::load(device.clone()).unwrap();
        let fs = fs_composition::load(device.clone()).unwrap();
        GraphicsPipeline::start()
            .vertex_input_state(gltf_loader::GltfVertex::per_vertex())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .render_pass(Subpass::from(render_pass, 0).unwrap())
            .build(device)
            .unwrap()
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

    fn create_framebuffers(images: &[Arc<SwapchainImage>], render_pass: Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
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

    fn create_composition_sets(&mut self, app: &App, gbuffer_pass: &GbufferPass, ssao_pass: &SsaoPass) {
        let regular_sampler = Sampler::new(app.device.clone(), SamplerCreateInfo {
            min_filter: Filter::Nearest,
            mag_filter: Filter::Nearest,
            ..SamplerCreateInfo::default()
        }).unwrap();

        // unfortunately we need a separate descriptor set for every framebuffer, because the
        // resulting image can be different for each renderpass and each framebuffer.
        self.composition_set = Some(gbuffer_pass.get_framebuffers().iter().enumerate().map(|(i, f): (usize, &Arc<Framebuffer>)| {
            PersistentDescriptorSet::new(
                &app.allocator_descriptor_set,
                self.pipeline.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::image_view_sampler(0, f.attachments()[1].clone(), regular_sampler.clone()), // position
                    WriteDescriptorSet::image_view_sampler(1, f.attachments()[2].clone(), regular_sampler.clone()), // normal
                    WriteDescriptorSet::image_view_sampler(2, f.attachments()[3].clone(), regular_sampler.clone()), // albedo
                    WriteDescriptorSet::image_view_sampler(3, ssao_pass.get_framebuffers()[i].attachments()[0].clone(), regular_sampler.clone()), // ssao
                    WriteDescriptorSet::image_view_sampler(4, f.attachments()[3].clone(), regular_sampler.clone()), // TODO: ssao blur
                    WriteDescriptorSet::buffer(5, self.ssao_settings.clone()),
                ]).unwrap()
        }).collect());
    }

    pub fn get_graphics_pipeline(&self) -> Arc<GraphicsPipeline> {
        self.pipeline.clone()
    }

    pub fn get_framebuffers(&self) -> Vec<Arc<Framebuffer>> {
        self.framebuffers.clone()
    }

    pub fn recreate_resources(&mut self, app: &App, viewport: &Viewport, gbuffer_pass: &GbufferPass, ssao_pass: &SsaoPass) {
        self.framebuffers = CompositionPass::create_framebuffers(&app.swapchain_images, self.render_pass.clone());
        self.pipeline = CompositionPass::create_graphics_pipeline(app.device.clone(), self.render_pass.clone(), viewport.clone());
        self.create_composition_sets(app, gbuffer_pass, ssao_pass);
    }
}
