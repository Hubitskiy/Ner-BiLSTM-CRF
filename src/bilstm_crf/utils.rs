use burn::prelude::*;

pub fn select_device() -> Device {
    #[cfg(target_os = "macos")]
    {
        println!("Metal backend was chosen");
        return Device::metal(Default::default());
    }

    #[cfg(any(target_os = "windows", target_os = "linux"))]
    {
        println!("WGPU backend was chosen");
        return Device::wgpu(Default::default());
    }

    #[allow(unreachable_code)]
    {
        println!("Default backend was chosen");
        Device::default()
    }
}
