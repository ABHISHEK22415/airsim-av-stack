#!/bin/bash
# Start the AirSim "AirSimNH" neighbourhood fullscreen on the NVIDIA GPU (HDMI TV).
DIR="$(cd "$(dirname "$0")" && pwd)"
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export DISPLAY=:0 XAUTHORITY=$(ls /run/user/1000/.mutter-Xwaylandauth.* | head -1)
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only
if pgrep -f "^$DIR/envs/.*/AirSimNH" >/dev/null; then echo "AirSim is already running"; exit 1; fi
exec "$DIR/envs/AirSimNH/LinuxNoEditor/AirSimNH.sh" -settings="$DIR/${SIM_SETTINGS:-settings_car.json}" \
  -ResX=1920 -ResY=1080 -windowed -ExecCmds="scalability 2" "$@"
