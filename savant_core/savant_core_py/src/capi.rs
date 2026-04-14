pub mod frame;
pub mod object;
pub mod pipeline;

use std::ffi::{c_char, CStr};

/// # Safety
///
/// - `external_version` must be a non-null pointer to a valid NUL-terminated C string that
///   remains readable for the duration of this call.
/// - The string must be valid UTF-8.
///
/// Returns `false` if the pointer is null or the string is not valid UTF-8.
#[no_mangle]
pub unsafe extern "C" fn check_version(external_version: *const c_char) -> bool {
    if external_version.is_null() {
        return false;
    }
    // SAFETY: caller guarantees non-null, NUL-terminated, valid for the duration of the call.
    let external_version = CStr::from_ptr(external_version);
    match external_version.to_str() {
        Ok(s) => savant_core::version() == s,
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::{c_char, CString};

    #[test]
    fn test_check_version() {
        unsafe {
            let ver = CString::new(savant_core::version()).unwrap();
            assert!(crate::capi::check_version(ver.as_ptr()));
        }
    }

    #[test]
    fn test_check_version_null_returns_false() {
        unsafe {
            assert!(!crate::capi::check_version(std::ptr::null()));
        }
    }

    #[test]
    fn test_check_version_invalid_utf8_returns_false() {
        let bad = [0xffu8, 0xfeu8, 0u8];
        unsafe {
            assert!(!crate::capi::check_version(bad.as_ptr().cast::<c_char>()));
        }
    }
}
