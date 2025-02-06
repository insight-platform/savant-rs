use gst::{glib, MetaAPI, MetaAPIExt};
use std::{fmt, mem};

#[repr(transparent)]
pub(crate) struct SavantFrameBatchIdMeta(imp::SavantFrameBatchIdMeta);

impl SavantFrameBatchIdMeta {
    pub fn ids(&self) -> &[i64] {
        &self.0.ids
    }

    pub fn replace(
        buffer: &mut gst::BufferRef,
        ids: Vec<i64>,
    ) -> gst::MetaRefMut<Self, gst::meta::Standalone> {
        unsafe {
            // Manually dropping because gst_buffer_add_meta() takes ownership of the
            // content of the struct
            let mut params = mem::ManuallyDrop::new(imp::SavantFrameBatchIdMetaParams { ids });

            let meta = gst::ffi::gst_buffer_add_meta(
                buffer.as_mut_ptr(),
                imp::savant_attribute_meta_get_info(),
                &mut *params as *mut imp::SavantFrameBatchIdMetaParams as glib::ffi::gpointer,
            ) as *mut imp::SavantFrameBatchIdMeta;

            Self::from_mut_ptr(buffer, meta)
        }
    }
}

unsafe impl Send for SavantFrameBatchIdMeta {}
unsafe impl Sync for SavantFrameBatchIdMeta {}

unsafe impl MetaAPI for SavantFrameBatchIdMeta {
    type GstType = imp::SavantFrameBatchIdMeta;

    fn meta_api() -> glib::Type {
        imp::savant_attribute_meta_api_get_type()
    }
}

impl fmt::Debug for SavantFrameBatchIdMeta {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SavantFrameBatchIdMeta")
            .field("ids", &self.0.ids)
            .finish()
    }
}

mod imp {
    use gst::glib;
    use gst::glib::translate::{from_glib, IntoGlib};
    use pyo3::ffi::c_str;
    use std::ptr;
    use std::sync::LazyLock;

    pub(super) struct SavantFrameBatchIdMetaParams {
        pub ids: Vec<i64>,
    }

    #[repr(C)]
    pub(crate) struct SavantFrameBatchIdMeta {
        parent: gst::ffi::GstMeta,
        pub(super) ids: Vec<i64>,
    }

    pub(super) fn savant_attribute_meta_api_get_type() -> glib::Type {
        static TYPE: LazyLock<glib::Type> = LazyLock::new(|| unsafe {
            let t = from_glib(gst::ffi::gst_meta_api_type_register(
                c_str!("GstSavantFrameBatchIdMetaAPI").as_ptr() as *const _,
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

        let meta = &mut *(meta as *mut SavantFrameBatchIdMeta);
        let params = ptr::read(params as *const SavantFrameBatchIdMetaParams);

        ptr::write(&mut meta.ids, params.ids);

        true.into_glib()
    }

    unsafe extern "C" fn savant_attribute_meta_free(
        meta: *mut gst::ffi::GstMeta,
        _buffer: *mut gst::ffi::GstBuffer,
    ) {
        let meta = &mut *(meta as *mut SavantFrameBatchIdMeta);

        ptr::drop_in_place(&mut meta.ids);
    }

    unsafe extern "C" fn savant_attribute_meta_transform(
        dest: *mut gst::ffi::GstBuffer,
        meta: *mut gst::ffi::GstMeta,
        _buffer: *mut gst::ffi::GstBuffer,
        _type_: glib::ffi::GQuark,
        _data: glib::ffi::gpointer,
    ) -> glib::ffi::gboolean {
        let meta = &*(meta as *mut SavantFrameBatchIdMeta);

        super::SavantFrameBatchIdMeta::replace(
            gst::BufferRef::from_mut_ptr(dest),
            meta.ids.clone(),
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
                    c_str!("GstSavantFrameBatchIdMeta").as_ptr() as *const _,
                    size_of::<SavantFrameBatchIdMeta>(),
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
