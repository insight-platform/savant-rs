use std::env;
use std::process::Command;

fn main() {
    let target = env::var("TARGET").unwrap();

    if target.contains("x86") {
        let mut cxx_flags = String::new();
        cxx_flags.push_str(" -mpclmul");
        cxx_flags.push_str(" -msse");
        println!("cargo:rustc-env=CXXFLAGS={}", cxx_flags.trim());
    }
}
