//! What a Vulkan pipeline actually costs to create, with and without a cache.
//!
//! This is the measurement behind the whole cache subsystem: if creating a pipeline were free,
//! there would be no reason to persist anything, and Citron's `invalidate ALL shader cache`
//! releases would be harmless. They are not.
//!
//! ```sh
//! # one .spv per pipeline variant, built with glslc -DK=<n>
//! cargo run --release --example vk_pipeline -- k1.spv k4.spv k16.spv k64.spv
//! ```
//!
//! Three numbers come out:
//!
//! - **cold** — `vkCreateComputePipelines` with no cache at all
//! - **warm** — the same pipelines again, from a `VkPipelineCache` serialized after the cold pass
//! - **per pipeline** — so the figure can be compared against a shader compile budget
//!
//! The device can be chosen with `--device <substring>`, which matters here: this box has three
//! AMD cards and the number is very different on a Vega 10 than on an RDNA4 part.

use std::ffi::{CStr, c_char};
use std::time::Instant;

use ash::vk;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut device_filter = String::new();
    let mut save: Option<String> = None;
    let mut load: Option<String> = None;
    let mut shaders = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--device" => device_filter = args.next().unwrap_or_default(),
            "--save" => save = args.next(),
            "--load" => load = args.next(),
            _ => shaders.push(a),
        }
    }
    if shaders.is_empty() {
        eprintln!("usage: vk_pipeline [--device <substr>] [--save <file> | --load <file>] <file.spv>...");
        std::process::exit(2);
    }

    let spirv: Vec<Vec<u32>> = shaders
        .iter()
        .map(|p| {
            let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
            assert!(bytes.len() % 4 == 0, "SPIR-V length is not a multiple of 4");
            bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect()
        })
        .collect();

    // SAFETY: the Vulkan loader is present; every handle created here is destroyed before return.
    let (entry, instance, _physical, device, _queue_family) = unsafe { init(&device_filter) };

    let device_name = unsafe {
        let props = instance.get_physical_device_properties(_physical);
        CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .into_owned()
    };
    println!("device: {device_name}");
    println!("pipelines: {}", spirv.len());

    // A persisted cache is built in one process and consumed in another. Measuring it inside a
    // single process measures radv's in-memory cache and reports a speedup of several hundred
    // times, which is true and useless.
    let initial: Vec<u8> = match &load {
        Some(path) => std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}")),
        None => Vec::new(),
    };
    let cache = unsafe {
        let ci = vk::PipelineCacheCreateInfo::default();
        let ci = if initial.is_empty() {
            ci
        } else {
            ci.initial_data(&initial)
        };
        device.create_pipeline_cache(&ci, None).expect("cache")
    };

    let (total, per) = unsafe { create_all(&device, &spirv, Some(cache)) };
    let label = if load.is_some() { "loaded" } else { "cold  " };
    println!("{label}: {total:>8.3} ms total, {per:>8.3} ms per pipeline");

    let blob = unsafe { device.get_pipeline_cache_data(cache) }.expect("cache data");
    println!("cache blob: {} bytes", blob.len());
    if let Some(path) = &save {
        std::fs::write(path, &blob).unwrap_or_else(|e| panic!("{path}: {e}"));
        println!("saved to {path}");
    }

    // A cache built from a corrupted blob: the spec allows the driver to accept it and simply
    // ignore the contents, so "no error" does NOT mean "valid". That is why the Rust cache module
    // validates structure itself instead of trusting the driver to reject damage.
    if !blob.is_empty() {
        let mut bad = blob.clone();
        for b in bad.iter_mut().take(16) {
            *b ^= 0xFF;
        }
        let ci = vk::PipelineCacheCreateInfo::default().initial_data(&bad);
        match unsafe { device.create_pipeline_cache(&ci, None) } {
            Ok(c) => {
                println!("corrupt blob: accepted without error (driver ignores the contents)");
                unsafe { device.destroy_pipeline_cache(c, None) };
            }
            Err(e) => println!("corrupt blob: rejected as {e:?}"),
        }
    }

    unsafe {
        device.destroy_pipeline_cache(cache, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
        drop(entry);
    }
}

unsafe fn init(
    filter: &str,
) -> (ash::Entry, ash::Instance, vk::PhysicalDevice, ash::Device, u32) {
    // `loaded` feature: resolve the loader at runtime rather than link-time.
    let entry = ash::Entry::load().expect("load libvulkan");
    let app = vk::ApplicationInfo::default()
        .application_name(c"vitrine")
        .api_version(vk::make_api_version(0, 1, 1, 0));
    let ci = vk::InstanceCreateInfo::default().application_info(&app);

    let instance = entry.create_instance(&ci, None).expect("instance");

    let devices = instance.enumerate_physical_devices().expect("devices");
    let physical = devices
        .iter()
        .copied()
        .find(|d| {
            let props = instance.get_physical_device_properties(*d);
            let name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
            filter.is_empty() || name.to_lowercase().contains(&filter.to_lowercase())
        })
        .unwrap_or_else(|| panic!("no device matching {filter:?}"));

    let props = instance.get_physical_device_properties(physical);
    let queue_family = instance
        .get_physical_device_queue_family_properties(physical)
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .expect("a compute queue") as u32;

    let priority = [1.0f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priority)];
    let dci = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
    let device = instance.create_device(physical, &dci, None).expect("device");
    let _ = props;
    (entry, instance, physical, device, queue_family)
}

/// Create every pipeline once, optionally from a cache, and return (total ms, per-pipeline ms).
unsafe fn create_all(
    device: &ash::Device,
    spirv: &[Vec<u32>],
    cache: Option<vk::PipelineCache>,
) -> (f64, f64) {
    let mut modules = Vec::new();
    for code in spirv {
        let ci = vk::ShaderModuleCreateInfo::default().code(code);
        modules.push(device.create_shader_module(&ci, None).expect("module"));
    }
    // The shader declares `binding 0` as a storage buffer, so the layout has to declare it too.
    // An empty layout with a shader that uses a binding is not merely sloppy: radv walks off the
    // end of the descriptor set layouts when it validates the interface and takes the process down.
    let binding = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];
    let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding);
    let set_layout = device
        .create_descriptor_set_layout(&dsl_ci, None)
        .expect("set layout");
    let layouts = [set_layout];
    let layout_ci = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
    let layout = device.create_pipeline_layout(&layout_ci, None).expect("layout");

    let stages: Vec<vk::PipelineShaderStageCreateInfo> = modules
        .iter()
        .map(|m| {
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(*m)
                .name(c"main")
        })
        .collect();

    let cache = cache.unwrap_or(vk::PipelineCache::null());
    let infos: Vec<vk::ComputePipelineCreateInfo> = stages
        .iter()
        .map(|s| {
            vk::ComputePipelineCreateInfo::default()
                .stage(*s)
                .layout(layout)
        })
        .collect();

    let start = Instant::now();
    let pipelines = device
        .create_compute_pipelines(cache, &infos, None)
        .expect("create pipelines");
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    for p in &pipelines {
        device.destroy_pipeline(*p, None);
    }
    for m in &modules {
        device.destroy_shader_module(*m, None);
    }
    device.destroy_pipeline_layout(layout, None);
    device.destroy_descriptor_set_layout(set_layout, None);

    (elapsed, elapsed / spirv.len() as f64)
}

/// Silence the unused-import warning for `c_char` when built on a platform that does not need it.
const _: Option<c_char> = None;
