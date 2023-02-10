pub mod camera;
pub mod gltf_loader;

use std::io::{Cursor, Read};
use std::sync::Arc;
use ktx::KtxInfo;

use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract};
use vulkano::descriptor_set::allocator::{StandardDescriptorSetAllocator};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageAccess, ImageDimensions, ImageUsage, ImmutableImage, MipmapsCount, SwapchainImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::sampler::{Sampler, SamplerCreateInfo};
use vulkano::swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo};
use vulkano::sync::GpuFuture;
use vulkano::VulkanLibrary;
use vulkano_win::VkSurfaceBuild;
use winit::dpi::{PhysicalSize, Size};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

pub const X_SIZE: u32 = 1280;
pub const Y_SIZE: u32 = 1024;

pub struct App {
    pub device: Arc<Device>,
    pub instance: Arc<Instance>,
    pub queue: Arc<Queue>,
    pub queue_family_index: u32,
    pub surface: Arc<Surface>,
    pub swapchain: Arc<Swapchain>,
    pub swapchain_images: Vec<Arc<SwapchainImage>>,
    pub allocator_command_buffer: Arc<StandardCommandBufferAllocator>,
    pub allocator_descriptor_set: Arc<StandardDescriptorSetAllocator>,
    pub allocator_memory: Arc<StandardMemoryAllocator>,
}

impl App {
    pub fn new(title: &str) -> (Self, EventLoop<()>) {
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let required_extensions = vulkano_win::required_extensions(library.as_ref());
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
            .expect("failed to create instance");

        let event_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(Size::Physical(PhysicalSize::new(X_SIZE, Y_SIZE)))
            .build_vk_surface(&event_loop, instance.clone())
            .expect("failed to create surface");

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_separate_depth_stencil_layouts: true,
            ..Default::default()
        };
        let (physical, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|d| d.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .expect("no suitable physical device found");
        println!("device: {}", physical.properties().device_name);

        println!("queue_family_index: {}", queue_family_index);

        let (device, mut queues) = Device::new(
            physical.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                enabled_features: Features {
                    separate_depth_stencil_layouts: true,
                    ..Features::default()
                },
                ..Default::default()
            },
        )
            .expect("failed to create device");

        let queue = queues.next().unwrap();

        // swapchain
        let caps = physical
            .surface_capabilities(&surface, Default::default())
            .expect("failed to get surface capabilities");

        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let image_format = Some(
            physical
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count + 1,
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha,
                present_mode: PresentMode::FifoRelaxed,
                ..Default::default()
            },
        )
            .unwrap();

        let allocator_command_buffer = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        let allocator_descriptor_set = Arc::new(StandardDescriptorSetAllocator::new(device.clone()));

        let allocator_memory = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        (App {
            allocator_command_buffer,
            allocator_descriptor_set,
            allocator_memory,
            device,
            instance,
            surface,
            swapchain,
            swapchain_images: images,
            queue,
            queue_family_index,
        }, event_loop)
    }

    pub fn get_framebuffers(&self, memory_allocator: &Arc<StandardMemoryAllocator>, images: &[Arc<SwapchainImage>], render_pass: &Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
        images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                let depth_buffer = ImageView::new_default(AttachmentImage::transient(memory_allocator, image.dimensions().width_height(), Format::D24_UNORM_S8_UINT).unwrap()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view, depth_buffer],
                        ..Default::default()
                    },
                )
                    .unwrap()
            })
            .collect::<Vec<_>>()
    }

    pub fn load_texture(&self, image_bytes: &[u8], width: u32, height: u32, mipmaps: u32, format: Format) -> Arc<ImmutableImage> {
        let mut uploads = AutoCommandBufferBuilder::primary(
            &self.allocator_command_buffer,
            self.queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
            .unwrap();
        let texture = {
            let dimensions = ImageDimensions::Dim2d {
                width,
                height,
                array_layers: 1,
            };
            ImmutableImage::from_iter(
                &self.allocator_memory,
                image_bytes.to_owned(),
                dimensions,
                MipmapsCount::Specific(mipmaps),
                format,
                &mut uploads,
            )
                .unwrap()
        };

        let _ = uploads
            .build()
            .unwrap()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush();

        texture
    }

    pub fn load_texture_ktx(&self, bytes: &[u8]) -> Arc<ImmutableImage> {
        let cursor = Cursor::new(bytes);
        let decoder = ktx::Decoder::new(cursor).unwrap();
        let mips = decoder.mipmap_levels();
        let width = decoder.pixel_width();
        let height = decoder.pixel_height();
        let image_data = decoder.read_textures().next().unwrap();
        self.load_texture(&image_data, width, height, mips, Format::R8G8B8A8_UNORM)
    }
}
