#!/usr/bin/env bash
# Reference trace: nvstreammux -> nvtracker -> fakesink (R&D / comparison with manual NvTracker path).
# Run on a DeepStream machine with IOU tracker assets. Adjust paths and live sources as needed.
#
# Example (synthetic): two test patterns -> mux -> tracker.
# Log batch meta via `GST_DEBUG` or attach a probe in Rust; this script is a shell baseline.

set -euo pipefail

LL_LIB="${LL_LIB:-/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so}"
CFG="${CFG:-$(dirname "$0")/../assets/config_tracker_IOU.yml}"

echo "LL_LIB=$LL_LIB"
echo "CFG=$CFG"
echo "Uncomment and edit the gst-launch line below for your environment."

# gst-launch-1.0 -e \
#   nvvideotestsrc pattern=0 ! nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA,width=320,height=240' ! mux.sink_0 \
#   nvvideotestsrc pattern=1 ! nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA,width=320,height=240' ! mux.sink_1 \
#   nvstreammux name=mux batch-size=2 width=320 height=240 ! \
#   nvtracker ll-lib-file="$LL_LIB" ll-config-file="$CFG" tracker-width=320 tracker-height=240 gpu-id=0 ! \
#   fakesink sync=false

exit 0
