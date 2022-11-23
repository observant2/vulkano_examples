use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::sync::GpuFuture;
use vulkano::{sync, VulkanLibrary};

pub fn main() {
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let instance =
        Instance::new(library, InstanceCreateInfo::default()).expect("failed to create instance");

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
        .expect("couldn't find a graphical queue family") as u32;

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

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let source_content: Vec<i32> = (0..64).collect();
    let source = CpuAccessibleBuffer::from_iter(
        memory_allocator.as_ref(),
        BufferUsage {
            transfer_src: true,
            ..Default::default()
        },
        false,
        source_content,
    )
    .expect("failed to create source buffer");

    let destination_content: Vec<i32> = (0..64).map(|_| 0).collect();
    let destination = CpuAccessibleBuffer::from_iter(
        memory_allocator.as_ref(),
        BufferUsage {
            transfer_dst: true,
            ..Default::default()
        },
        false,
        destination_content,
    )
    .expect("failed to create destination buffer");

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.as_ref(),
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("failed to create CommandBufferBuilder");

    builder
        .copy_buffer(CopyBufferInfo::buffers(source.clone(), destination.clone()))
        .unwrap();

    let command_buffer = builder.build().expect("failed to build command_buffer");

    {
        let src_content = source.read().unwrap();
        let destination_content = destination.read().unwrap();

        println!(
            "src: {:?}, dest: {:?}",
            src_content.to_vec(),
            destination_content.to_vec()
        );
    }

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    {
        let src_content = source.read().unwrap();
        let destination_content = destination.read().unwrap();

        println!(
            "src: {:?}, dest: {:?}",
            src_content.to_vec(),
            destination_content.to_vec()
        );
    }
}
