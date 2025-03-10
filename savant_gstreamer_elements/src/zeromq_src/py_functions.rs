use anyhow::bail;
use gst::prelude::*;
use gst::subclass::prelude::*;
use pyo3::prelude::*;
use savant_core::message::Message;
use savant_core::primitives::rust::VideoFrameProxy;
use savant_core_py::gst::{InvocationReason, REGISTERED_HANDLERS};
use savant_core_py::primitives::message::Message as PyMessage;
use std::borrow::Cow;

use crate::zeromq_src::CAT;

use super::ZeromqSrc;

impl ZeromqSrc {
    pub(crate) fn invoke_custom_transform_py_function_on_message<'a>(
        &'_ self,
        message: &'a Message,
    ) -> anyhow::Result<Cow<'a, Message>> {
        if self.settings.lock().invoke_on_message {
            let res = Python::with_gil(|py| {
                let element_name = self.obj().name().to_string();
                let handlers_bind = REGISTERED_HANDLERS.read();
                let handler = handlers_bind.get(&element_name);

                if let Some(handler) = handler {
                    let message = message.clone();
                    let py_message = PyMessage::new(message);
                    let res = handler.call1(
                        py,
                        (
                            element_name,
                            InvocationReason::IngressMessageTransformer,
                            py_message,
                        ),
                    );
                    match res {
                        Ok(res) => {
                            gst::trace!(CAT, imp = self, "Handler invoked successfully");
                            let message = res.extract::<PyMessage>(py);
                            match message {
                                Ok(message) => Ok(Cow::Owned(message.extract())),
                                Err(e) => {
                                    gst::error!(
                                        CAT,
                                        imp = self,
                                        "Handler invocation failed: {}",
                                        e
                                    );
                                    bail!("Handler invocation failed (cannot extract Message type): {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            gst::error!(CAT, imp = self, "Handler invocation failed: {}", e);
                            bail!("Handler invocation failed: {}", e);
                        }
                    }
                } else {
                    Ok(Cow::Borrowed(message))
                }
            })?;
            Ok(res)
        } else {
            Ok(Cow::Borrowed(message))
        }
    }

    pub(crate) fn invoke_custom_ingress_py_function_on_frame(
        &self,
        _frame: &VideoFrameProxy,
    ) -> anyhow::Result<bool> {
        Ok(true)
    }
}
