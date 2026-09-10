use std::{env, path::PathBuf};

fn cuda_path() -> PathBuf {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH_V13_1");
    println!("cargo:rerun-if-env-changed=CUDA_PATH_V13_0");
    println!("cargo:rerun-if-env-changed=CUDA_PATH_V12_9");
    println!("cargo:rerun-if-env-changed=CUDA_PATH_V12_8");

    for name in ["CUDA_PATH", "CUDA_PATH_V13_1", "CUDA_PATH_V13_0", "CUDA_PATH_V12_9", "CUDA_PATH_V12_8"] {
        if let Ok(value) = env::var(name) {
            let path = PathBuf::from(value);
            if path.exists() {
                return path;
            }
        }
    }

    if !cfg!(target_os = "windows") {
        let conventional = PathBuf::from("/usr/local/cuda");
        if conventional.exists() {
            return conventional;
        }
    }

    panic!("CUDA Toolkit was not found; install it under /usr/local/cuda or set CUDA_PATH to its install directory");
}

fn main() {
    let cuda = cuda_path();
    let include = cuda.join("include");
    let lib_dir = if cfg!(target_os = "windows") { cuda.join("lib/x64") } else { cuda.join("lib64") };
    let nvcc = cuda.join("bin/nvcc");

    if !include.exists() {
        panic!("CUDA include directory does not exist: {}", include.display());
    }
    if !lib_dir.exists() {
        panic!("CUDA library directory does not exist: {}", lib_dir.display());
    }

    println!("cargo:rerun-if-changed=cpp/bulletou_cuda_backend.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");

    let mut build = cc::Build::new();
    build.cuda(true);
    if env::var_os("NVCC").is_none() && nvcc.is_file() {
        // SAFETY: Cargo build scripts are separate, single-threaded processes;
        // this only selects cc-rs' CUDA compiler before compilation starts.
        unsafe { env::set_var("NVCC", nvcc) };
    }
    build.cudart("shared");
    build.cargo_warnings(false);
    build.include(include);
    build.file("cpp/bulletou_cuda_backend.cu");
    build.flag("-std=c++17");
    build.flag("-O3");
    build.flag("--use_fast_math");
    build.compile("bulletou_cuda_cpp_backend");
}
