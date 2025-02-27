use gst::{glib, MetaAPI, MetaAPIExt};
use std::{fmt, mem};

#[repr(transparent)]
pub struct SavantIdMeta(imp::SavantIdMeta);

impl SavantIdMeta {
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
            let mut params = mem::ManuallyDrop::new(imp::SavantIdMetaParams { ids });

            let meta = gst::ffi::gst_buffer_add_meta(
                buffer.as_mut_ptr(),
                imp::savant_id_meta_get_info(),
                &mut *params as *mut imp::SavantIdMetaParams as glib::ffi::gpointer,
            ) as *mut imp::SavantIdMeta;

            Self::from_mut_ptr(buffer, meta)
        }
    }
}

unsafe impl Send for SavantIdMeta {}
unsafe impl Sync for SavantIdMeta {}

unsafe impl MetaAPI for SavantIdMeta {
    type GstType = imp::SavantIdMeta;

    fn meta_api() -> glib::Type {
        imp::savant_id_meta_api_get_type()
    }
}

impl fmt::Debug for SavantIdMeta {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SavantIdMeta")
            .field("ids", &self.0.ids)
            .finish()
    }
}

mod imp {
    use gst::glib;
    use gst::glib::translate::{from_glib, IntoGlib};
    use std::ptr;
    use std::sync::LazyLock;

    pub struct SavantIdMetaParams {
        pub ids: Vec<i64>,
    }

    #[repr(C)]
    pub struct SavantIdMeta {
        parent: gst::ffi::GstMeta,
        pub(super) ids: Vec<i64>,
    }

    pub(super) fn savant_id_meta_api_get_type() -> glib::Type {
        static TYPE: LazyLock<glib::Type> = LazyLock::new(|| unsafe {
            let t = from_glib(gst::ffi::gst_meta_api_type_register(
                c"GstSavantIdMetaAPI".as_ptr() as *const _,
                [ptr::null::<std::os::raw::c_char>()].as_ptr() as *mut *const _,
            ));

            assert_ne!(t, glib::Type::INVALID);

            t
        });

        *TYPE
    }

    unsafe extern "C" fn savant_id_meta_init(
        meta: *mut gst::ffi::GstMeta,
        params: glib::ffi::gpointer,
        _buffer: *mut gst::ffi::GstBuffer,
    ) -> glib::ffi::gboolean {
        assert!(!params.is_null());

        let meta = &mut *(meta as *mut SavantIdMeta);
        let params = ptr::read(params as *const SavantIdMetaParams);

        ptr::write(&mut meta.ids, params.ids);

        true.into_glib()
    }

    unsafe extern "C" fn savant_id_meta_free(
        meta: *mut gst::ffi::GstMeta,
        _buffer: *mut gst::ffi::GstBuffer,
    ) {
        let meta = &mut *(meta as *mut SavantIdMeta);

        ptr::drop_in_place(&mut meta.ids);
    }

    unsafe extern "C" fn savant_id_meta_transform(
        dest: *mut gst::ffi::GstBuffer,
        meta: *mut gst::ffi::GstMeta,
        _buffer: *mut gst::ffi::GstBuffer,
        _type_: glib::ffi::GQuark,
        _data: glib::ffi::gpointer,
    ) -> glib::ffi::gboolean {
        let meta = &*(meta as *mut SavantIdMeta);

        super::SavantIdMeta::replace(gst::BufferRef::from_mut_ptr(dest), meta.ids.clone());

        true.into_glib()
    }

    pub(super) unsafe fn savant_id_meta_get_info() -> *const gst::ffi::GstMetaInfo {
        struct MetaInfo(ptr::NonNull<gst::ffi::GstMetaInfo>);
        unsafe impl Send for MetaInfo {}
        unsafe impl Sync for MetaInfo {}

        static META_INFO: LazyLock<MetaInfo> = LazyLock::new(|| unsafe {
            MetaInfo(
                ptr::NonNull::new(gst::ffi::gst_meta_register(
                    savant_id_meta_api_get_type().into_glib(),
                    c"GstSavantIdMeta".as_ptr() as *const _,
                    size_of::<SavantIdMeta>(),
                    Some(savant_id_meta_init),
                    Some(savant_id_meta_free),
                    Some(savant_id_meta_transform),
                ) as *mut gst::ffi::GstMetaInfo)
                .expect("Failed to register meta API"),
            )
        });

        META_INFO.0.as_ptr()
    }
}
