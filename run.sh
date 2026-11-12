#!/bin/bash
# Launch the drone animation fullscreen, rendered on the NVIDIA GPU
export XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only
export DISPLAY=:0 XAUTHORITY=$(ls /run/user/1000/.mutter-Xwaylandauth.* | head -1)
exec google-chrome --user-data-dir="$HOME/.config/drone-animation-chrome" \
  --ozone-platform=x11 --ignore-gpu-blocklist --enable-gpu-rasterization --enable-zero-copy \
  --use-gl=angle --use-angle=gl --disable-features=Vulkan --force_high_performance_gpu --no-first-run --noerrdialogs \
  --disable-session-crashed-bubble --start-fullscreen --kiosk \
  --app="file://$HOME/drone-animation/index.html" "$@"
