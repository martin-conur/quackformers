fn main() {
    // Detect platform
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-cfg=use_accelerate");
        println!("cargo:rustc-cfg=feature=\"accelerate\"");
    }
    
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-cfg=use_mkl");
        println!("cargo:rustc-cfg=feature=\"mkl\"");
    }
    
    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-cfg=use_mkl");
        println!("cargo:rustc-cfg=feature=\"mkl\"");
    }
}