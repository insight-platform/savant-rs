We are building a new library picasso which is built on top of Rust deepstream_encoders and deepstream_nvbufsurface and provides Python and Rust API.

The goal of the library is to receive savant-rs VideoFrame and Nvidia GPU-allocated GstBuffer pairs (I will call it a Frame below) or EOS for multiple sources, spawn NvEncoders internally and pass input pairs into output pairs and EOS messages. 

When Frame is received the Encoder uses specification to decide how to process it. 

There are three four scenarios:

1. drop (default when no conf for a stream);
2. bypass: buffer is dropped, VideoFrame goes to the output (when the codec is configured to copy specification, in this mode nothing related to 3,4 happens).
3. encode: buffer is copied to output buffer according to the conversion specification (TransformConfig), on this step the destination frame is available to user code in the form of GpuMat for basic graphic operations;
4. render and encode: it is copied to OpenGL canvas where Skia draws according to the specification, then optional user callback (Rust and Python) can be invoked to do custom render, and finally it goes to (2).

Three scenarious (2,3,4) support conditional processing, which is managed by a presence of particular attribute set for a frame, or if not set, all frames are processed.

For every source_id, user can define individual specification and change the specification later. Specification change concerning codec, resolution or transformation changes require the library to handle currently served frames and re-create the encoder to reflect specification changes. 

The library must support non-blocking operation, it means that every source is served in a separate thread and has individual queue for sending data into it. When the user sent Frame he is free to send another one.

The library must define max idle timeout for stream, so when no data arrived there, the library must invoke a special callback handler (Rust and Python respectively) to request what to do: keep it for X seconds or terminate it and send EOS or not.

When EOS is sent for a stream, it must send ensure the current encoder completed currently processed and awaiting Frames and propagate EOS to the output. Next Frame will cause Encoder recreation.

Callback summary:
- on_render (optional, invoked when user wants use Skia from Python or Rust to render their figures on Skia canvas before copying it to the destination buffer);
- on_object_draw_spec (optinal, invoked for every VideoObject and allow either update the spec defined for such stream/ns/label or define the spec);
- on_gpumat (optional, invoked when a destination buffer is created and access can be provided to NvBufSurface in the form of GpuMat);
- on_eviction(optional, invoked when the source must be evicted but the system want user approval/prolongation);

Specifications (for a source):
- conditional frame processing based on presence of a configured in-frame attribute (if none, all frames are processed);
- conditional video rendering based on presence of in-frame attribute (if none, all frames are rendered);
- object rendering specification (ObjectDraw defined for {(ns, labels), ...]: ObjectDraw}); In ObjectDraw, LabelDraw defines a pattern allowing interpolation of VideoObject fields like id, ns, label, conf, box, tracking info, etc; 
- on_render invocation for source (same callback for all sources);
- on_gpumat invocation for source (same callback for all sources);
- destination frame geometry specification (extend to allow extra paddings on the right/top/left/bottom);
- codec specification (drop (1), bypass (2), encode(3));

Specifications (general):
- on_eviction invocation for sources;

Per-stream specifications can be changed at any moment causing (when needed) internal reconfiguration as previously discussed.