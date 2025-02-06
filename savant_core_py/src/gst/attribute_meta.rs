use gst::{glib, MetaAPI, MetaAPIExt};
use savant_core::primitives::Attribute;
use std::{fmt, mem};

#[repr(transparent)]
pub(crate) struct SavantAttributeMeta(imp::SavantAttributeMeta);

impl SavantAttributeMeta {
    pub fn attributes(&self) -> &[Attribute] {
        &self.0.attributes
    }

    pub fn replace(
        buffer: &mut gst::BufferRef,
        attributes: Vec<Attribute>,
    ) -> gst::MetaRefMut<Self, gst::meta::Standalone> {
        unsafe {
            // Manually dropping because gst_buffer_add_meta() takes ownership of the
            // content of the struct
            let mut params = mem::ManuallyDrop::new(imp::SavantAttributeMetaParams { attributes });

            let meta = gst::ffi::gst_buffer_add_meta(
                buffer.as_mut_ptr(),
                imp::savant_attribute_meta_get_info(),
                &mut *params as *mut imp::SavantAttributeMetaParams as glib::ffi::gpointer,
            ) as *mut imp::SavantAttributeMeta;

            Self::from_mut_ptr(buffer, meta)
        }
    }
}

unsafe impl Send for SavantAttributeMeta {}
unsafe impl Sync for SavantAttributeMeta {}

unsafe impl MetaAPI for SavantAttributeMeta {
    type GstType = imp::SavantAttributeMeta;

    fn meta_api() -> glib::Type {
        imp::savant_attribute_meta_api_get_type()
    }
}

impl fmt::Debug for SavantAttributeMeta {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SavantAttributeMeta")
            .field("attributes", &self.0.attributes)
            .finish()
    }
}

mod imp {
    use gst::glib;
    use gst::glib::translate::{from_glib, IntoGlib};
    use pyo3::ffi::c_str;
    use savant_core::primitives::Attribute;
    use std::ptr;
    use std::sync::LazyLock;

    pub(super) struct SavantAttributeMetaParams {
        pub attributes: Vec<Attribute>,
    }

    #[repr(C)]
    pub(crate) struct SavantAttributeMeta {
        parent: gst::ffi::GstMeta,
        pub(super) attributes: Vec<Attribute>,
    }

    pub(super) fn savant_attribute_meta_api_get_type() -> glib::Type {
        static TYPE: LazyLock<glib::Type> = LazyLock::new(|| unsafe {
            let t = from_glib(gst::ffi::gst_meta_api_type_register(
                c_str!("GstSavantAttributeMetaAPI").as_ptr() as *const _,
                [ptr::null::<std::os::raw::c_char>()].as_ptr() as *mut *const _,
            ));

            assert_ne!(t, glib::Type::INVALID);

            t
        });

        *TYPE
    }

    unsafe extern "C" fn savant_attribute_meta_init(
        meta: *mut gst::ffi::GstMeta,
        params: glib::ffi::gpointer,
        _buffer: *mut gst::ffi::GstBuffer,
    ) -> glib::ffi::gboolean {
        assert!(!params.is_null());

        let meta = &mut *(meta as *mut SavantAttributeMeta);
        let params = ptr::read(params as *const SavantAttributeMetaParams);

        ptr::write(&mut meta.attributes, params.attributes);

        true.into_glib()
    }

    unsafe extern "C" fn savant_attribute_meta_free(
        meta: *mut gst::ffi::GstMeta,
        _buffer: *mut gst::ffi::GstBuffer,
    ) {
        let meta = &mut *(meta as *mut SavantAttributeMeta);

        ptr::drop_in_place(&mut meta.attributes);
    }

    unsafe extern "C" fn savant_attribute_meta_transform(
        dest: *mut gst::ffi::GstBuffer,
        meta: *mut gst::ffi::GstMeta,
        _buffer: *mut gst::ffi::GstBuffer,
        _type_: glib::ffi::GQuark,
        _data: glib::ffi::gpointer,
    ) -> glib::ffi::gboolean {
        let meta = &*(meta as *mut SavantAttributeMeta);

        super::SavantAttributeMeta::replace(
            gst::BufferRef::from_mut_ptr(dest),
            meta.attributes.clone(),
        );

        true.into_glib()
    }

    pub(super) unsafe fn savant_attribute_meta_get_info() -> *const gst::ffi::GstMetaInfo {
        struct MetaInfo(ptr::NonNull<gst::ffi::GstMetaInfo>);
        unsafe impl Send for MetaInfo {}
        unsafe impl Sync for MetaInfo {}

        static META_INFO: LazyLock<MetaInfo> = LazyLock::new(|| unsafe {
            MetaInfo(
                ptr::NonNull::new(gst::ffi::gst_meta_register(
                    savant_attribute_meta_api_get_type().into_glib(),
                    c_str!("GstSavantAttributeMeta").as_ptr() as *const _,
                    size_of::<SavantAttributeMeta>(),
                    Some(savant_attribute_meta_init),
                    Some(savant_attribute_meta_free),
                    Some(savant_attribute_meta_transform),
                ) as *mut gst::ffi::GstMetaInfo)
                .expect("Failed to register meta API"),
            )
        });

        META_INFO.0.as_ptr()
    }
}
