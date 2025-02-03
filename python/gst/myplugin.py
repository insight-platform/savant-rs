from gi.repository import Gst, GObject

Gst.init(None)


class MyPythonFilter(Gst.Element):
    __gstmetadata__ = ("MyPythonFilter", "Filter", "Custom Python Filter", "Author")

    _src_template = Gst.PadTemplate.new(
        "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS,
        Gst.Caps.new_any()
    )
    _sink_template = Gst.PadTemplate.new(
        "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS,
        Gst.Caps.new_any()
    )

    __gsttemplates__ = (_src_template, _sink_template)

    def __init__(self):
        super(MyPythonFilter, self).__init__()
        self.sinkpad = Gst.Pad.new_from_template(self._sink_template, "sink")
        self.srcpad = Gst.Pad.new_from_template(self._src_template, "src")
        self.add_pad(self.sinkpad)
        self.add_pad(self.srcpad)

    def do_chain(self, buf):
        print("Processing buffer in Python plugin")
        return self.srcpad.push(buf)


GObject.type_register(MyPythonFilter)
Gst.Element.register(None, "mypythonfilter", 0, MyPythonFilter)
