use gstreamer::{glib, MetaAPI, MetaAPIExt};
use std::{fmt, mem};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SavantIdMetaKind {
    Frame(i64),
    Batch(i64),
}

#[repr(transparent)]
pub struct SavantIdMeta(imp::SavantIdMeta);

impl SavantIdMeta {
    pub fn ids(&self) -> &[SavantIdMetaKind] {
        &self.0.ids
    }

    pub fn replace(
        buffer: &mut gstreamer::BufferRef,
        ids: Vec<SavantIdMetaKind>,
    ) -> gstreamer::MetaRefMut<Self, gstreamer::meta::Standalone> {
        unsafe {
            // Manually dropping because gst_buffer_add_meta() takes ownership of the
            // content of the struct
            let mut params = mem::ManuallyDrop::new(imp::SavantIdMetaParams { ids });

            let meta = gstreamer::ffi::gst_buffer_add_meta(
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
    use gstreamer::glib;
    use gstreamer::glib::translate::{from_glib, IntoGlib};
    use std::mem::size_of;
    use std::ptr;
    use std::sync::LazyLock;

    use super::SavantIdMetaKind;

    pub struct SavantIdMetaParams {
        pub ids: Vec<SavantIdMetaKind>,
    }

    #[repr(C)]
    pub struct SavantIdMeta {
        parent: gstreamer::ffi::GstMeta,
        pub(super) ids: Vec<SavantIdMetaKind>,
    }

    pub(super) fn savant_id_meta_api_get_type() -> glib::Type {
        static TYPE: LazyLock<glib::Type> = LazyLock::new(|| unsafe {
            let t = from_glib(gstreamer::ffi::gst_meta_api_type_register(
                c"GstSavantIdMetaAPI".as_ptr() as *const _,
                [ptr::null::<std::os::raw::c_char>()].as_ptr() as *mut *const _,
            ));

            assert_ne!(t, glib::Type::INVALID);

            t
        });

        *TYPE
    }

    unsafe extern "C" fn savant_id_meta_init(
        meta: *mut gstreamer::ffi::GstMeta,
        params: glib::ffi::gpointer,
        _buffer: *mut gstreamer::ffi::GstBuffer,
    ) -> glib::ffi::gboolean {
        assert!(!params.is_null());

        let meta = &mut *(meta as *mut SavantIdMeta);
        let params = ptr::read(params as *const SavantIdMetaParams);

        ptr::write(&mut meta.ids, params.ids);

        true.into_glib()
    }

    unsafe extern "C" fn savant_id_meta_free(
        meta: *mut gstreamer::ffi::GstMeta,
        _buffer: *mut gstreamer::ffi::GstBuffer,
    ) {
        let meta = &mut *(meta as *mut SavantIdMeta);

        ptr::drop_in_place(&mut meta.ids);
    }

    unsafe extern "C" fn savant_id_meta_transform(
        dest: *mut gstreamer::ffi::GstBuffer,
        meta: *mut gstreamer::ffi::GstMeta,
        _buffer: *mut gstreamer::ffi::GstBuffer,
        _type_: glib::ffi::GQuark,
        _data: glib::ffi::gpointer,
    ) -> glib::ffi::gboolean {
        let meta = &*(meta as *mut SavantIdMeta);

        super::SavantIdMeta::replace(gstreamer::BufferRef::from_mut_ptr(dest), meta.ids.clone());

        true.into_glib()
    }

    pub(super) unsafe fn savant_id_meta_get_info() -> *const gstreamer::ffi::GstMetaInfo {
        struct MetaInfo(ptr::NonNull<gstreamer::ffi::GstMetaInfo>);
        unsafe impl Send for MetaInfo {}
        unsafe impl Sync for MetaInfo {}

        static META_INFO: LazyLock<MetaInfo> = LazyLock::new(|| unsafe {
            MetaInfo(
                ptr::NonNull::new(gstreamer::ffi::gst_meta_register(
                    savant_id_meta_api_get_type().into_glib(),
                    c"GstSavantIdMeta".as_ptr() as *const _,
                    size_of::<SavantIdMeta>(),
                    Some(savant_id_meta_init),
                    Some(savant_id_meta_free),
                    Some(savant_id_meta_transform),
                ) as *mut gstreamer::ffi::GstMetaInfo)
                .expect("Failed to register meta API"),
            )
        });

        META_INFO.0.as_ptr()
    }
}
