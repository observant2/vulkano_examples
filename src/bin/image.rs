use std::sync::Arc;

use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage};
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo};
use vulkano::format::{ClearColorValue, Format};
use vulkano::image::{ImageDimensions, StorageImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::sync::GpuFuture;
use vulkano::{sync, VulkanLibrary};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
    uint data[];
} buf;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    buf.data[idx] *= 12;
}"
    }
}

pub fn main() {
    let app = App::new();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(app.device.clone()));

    let image = StorageImage::new(
        memory_allocator.as_ref(),
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(app.queue_family_index),
    )
    .unwrap();

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        app.device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.as_ref(),
        app.queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("failed to create CommandBufferBuilder");

    builder
        .clear_color_image(ClearColorImageInfo {
            clear_value: ClearColorValue::Float([1., 0., 1., 1.]),
            ..ClearColorImageInfo::image(image.clone())
        })
        .expect("failed to bind descriptor sets");

    let command_buffer = Arc::new(builder.build().expect("failed to build command_buffer"));

    let future = sync::now(app.device.clone())
        .then_execute(app.queue.clone(), command_buffer.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();
}

struct App {
    device: Arc<Device>,
    queue: Arc<Queue>,
    queue_family_index: u32,
}

impl App {
    pub fn new() -> Self {
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let instance = Instance::new(library, InstanceCreateInfo::default())
            .expect("failed to create instance");

        let physical = instance
            .enumerate_physical_devices()
            .expect("could not enumerate devices")
            .next()
            .expect("no devices available");

        println!("device: {}", physical.properties().device_name);

        for family in physical.queue_family_properties() {
            println!(
                "Found a queue family with {} queue(s): {:?}",
                family.queue_count, family.queue_flags
            );
        }

        let queue_family_index = physical
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_, q)| q.queue_flags.graphics)
            .expect("couldn't find a graphical queue family")
            as u32;

        println!("queue_family_index: {}", queue_family_index);

        let (device, mut queues) = Device::new(
            physical,
            DeviceCreateInfo {
                // here we pass the desired queue family to use by index
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("failed to create device");

        let queue = queues.next().unwrap();

        App {
            device,
            queue,
            queue_family_index,
        }
    }
}
