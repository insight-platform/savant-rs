mod frame;
mod object;
pub mod pipeline;

use std::ffi::{c_char, CStr};

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn check_version(external_version: *const c_char) -> bool {
    let external_version = CStr::from_ptr(external_version);
    savant_core::version()
        == *external_version.to_str().expect(
            "Failed to convert external version to string. This is a bug. Please report it.",
        )
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    #[test]
    fn test_check_version() {
        unsafe {
            let ver = CString::new(savant_core::version()).unwrap();
            assert!(crate::capi::check_version(ver.as_ptr()));
        }
    }
}
