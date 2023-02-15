use std::sync::Arc;

use nalgebra_glm::{identity, normalize, Vec4, vec4};
use rand::Rng;
use vulkano::buffer::{Buffer, BufferAllocateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::{AttachmentImage, ImageAccess, ImageUsage, ImmutableImage, SwapchainImage};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{GraphicsPipeline, Pipeline};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};

use vulkano_examples::{App, gltf_loader};

use crate::gbuffer_pipeline::GbufferPass;
use crate::shaders::{fs_ssao, vs_fullscreen};

use crate::shaders::fs_ssao::ty::{Projection, SsaoKernel};

const SSAO_NOISE_DIM: usize = 4;

pub struct SsaoPass {
    pub ssao_set: Option<Vec<Arc<PersistentDescriptorSet>>>,
    pub projection_buffer: Subbuffer<Projection>,

    ssao_kernel: Subbuffer<SsaoKernel>,
    ssao_noise: Arc<ImageView<ImmutableImage>>,

    pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
}

fn create_ssao_kernel() -> SsaoKernel {
    let lerp = |a: f32, b: f32, f: f32|
        {
            a + f * (b - a)
        };

    let mut rnd = rand::thread_rng();

    // Sample kernel
    let mut samples = [Vec4::identity().data.0[0]; 64];
    for i in 0..64
    {
        // Generate random point in +z direction
        let mut sample = vec4(
            rnd.gen_range(-1.0..=1.0),
            rnd.gen_range(-1.0..=1.0),
            rnd.gen_range(0.0..=1.0),
            0.0);

        // normalize to make sure it's in a round hemisphere, not in a cuboid
        sample = normalize(&sample);

        // now more points are in the hemisphere border than in the hemisphere -> redistribute
        sample *= rnd.gen_range(0.0..=1.0);

        // sample points should cluster near the origin
        let scale = i as f32 / 64.0;
        sample *= lerp(0.1, 1.0, scale * scale);

        samples[i] = sample.data.0[0];
    }

    SsaoKernel {
        samples,
    }
}

fn create_noise_texture(app: &App) -> Arc<ImmutableImage> {
    let mut rnd = rand::thread_rng();

    let mut ssao_noise = [[0.0, 0.0, 0.0, 0.0]; SSAO_NOISE_DIM * SSAO_NOISE_DIM];
    for n in &mut ssao_noise
    {
        *n = vec4(rnd.gen_range(0.0..=1.0) as f32, rnd.gen_range(0.0..=1.0) as f32, 0.0, 0.0).data.0[0];
    }

    app.load_texture(bytemuck::cast_slice(&ssao_noise), SSAO_NOISE_DIM as u32, SSAO_NOISE_DIM as u32, 1, Format::R32G32B32A32_SFLOAT)
}

impl SsaoPass {
    pub fn new(app: &App, viewport: &Viewport, gbuffer_pass: &GbufferPass) -> Self {
        let projection_buffer = Buffer::from_data(
            &app.allocator_memory,
            BufferAllocateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..BufferAllocateInfo::default()
            },
            Projection { projection: identity().data.0 },
        ).unwrap();

        let ssao_kernel = Buffer::from_data(
            &app.allocator_memory,
            BufferAllocateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..BufferAllocateInfo::default()
            },
            create_ssao_kernel(),
        ).unwrap();

        let ssao_noise = ImageView::new_default(create_noise_texture(&app)).unwrap();

        let render_pass = SsaoPass::create_render_pass(&app);
        let pipeline = SsaoPass::create_graphics_pipeline(app.device.clone(), render_pass.clone(), viewport.clone());
        let framebuffers = SsaoPass::create_framebuffers(app.allocator_memory.clone(), &app.swapchain_images, render_pass.clone());

        let mut ret = Self {
            pipeline,
            render_pass,
            framebuffers,
            ssao_noise,
            ssao_kernel,
            projection_buffer,
            ssao_set: None,
        };

        ret.create_ssao_sets(app, gbuffer_pass);

        ret
    }

    fn create_graphics_pipeline(device: Arc<Device>, render_pass: Arc<RenderPass>, viewport: Viewport) -> Arc<GraphicsPipeline> {
        let vs = vs_fullscreen::load(device.clone()).unwrap();
        let fs = fs_ssao::load(device.clone()).unwrap();
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
            .build(device.clone())
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

    fn create_framebuffers(memory_allocator: Arc<StandardMemoryAllocator>, images: &[Arc<SwapchainImage>], render_pass: Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
        images
            .iter()
            .map(|image| {
                let color = ImageView::new_default(AttachmentImage::with_usage(
                    &memory_allocator,
                    image.dimensions().width_height(),
                    // because we only use the red component we could use a single component framebuffer here...
                    image.format(),
                    ImageUsage::SAMPLED | image.usage()).unwrap()).unwrap();

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

    fn create_ssao_sets(&mut self, app: &App, gbuffer_pass: &GbufferPass) {
        let regular_sampler = Sampler::new(app.device.clone(), SamplerCreateInfo {
            min_filter: Filter::Nearest,
            mag_filter: Filter::Nearest,
            ..SamplerCreateInfo::default()
        }).unwrap();
        let noise_sampler = Sampler::new(app.device.clone(), SamplerCreateInfo {
            min_filter: Filter::Nearest,
            mag_filter: Filter::Nearest,
            address_mode: [SamplerAddressMode::Repeat; 3],
            ..SamplerCreateInfo::default()
        }).unwrap();

        self.ssao_set = Some(gbuffer_pass.get_framebuffers().iter().enumerate().map(|(_i, f): (usize, &Arc<Framebuffer>)| {
            PersistentDescriptorSet::new(
                &app.allocator_descriptor_set,
                self.pipeline.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::image_view_sampler(0, f.attachments()[1].clone(), regular_sampler.clone()), // position
                    WriteDescriptorSet::image_view_sampler(1, f.attachments()[2].clone(), regular_sampler.clone()), // normal
                    WriteDescriptorSet::image_view_sampler(2, self.ssao_noise.clone(), noise_sampler.clone()), // ssao noise
                    WriteDescriptorSet::buffer(3, self.ssao_kernel.clone()),
                    WriteDescriptorSet::buffer(4, self.projection_buffer.clone()),
                ]).unwrap()
        }).collect());
    }

    pub fn get_graphics_pipeline(&self) -> Arc<GraphicsPipeline> {
        self.pipeline.clone()
    }

    pub fn get_framebuffers(&self) -> Vec<Arc<Framebuffer>> {
        self.framebuffers.clone()
    }

    pub fn get_descriptor_sets(&self) -> Vec<Arc<PersistentDescriptorSet>> {
        self.ssao_set.as_ref().unwrap().clone()
    }

    pub fn recreate_resources(&mut self, app: &App, viewport: &Viewport, gbuffer_pass: &GbufferPass) {
        self.pipeline = SsaoPass::create_graphics_pipeline(app.device.clone(), self.render_pass.clone(), viewport.clone());
        self.framebuffers = SsaoPass::create_framebuffers(app.allocator_memory.clone(), &app.swapchain_images, self.render_pass.clone());
        self.create_ssao_sets(app, gbuffer_pass);
    }
}
