#!/usr/bin/env python
"""Stream the AV demo to a browser (e.g. the Mac over Tailscale) as MJPEG.

  /            page with both views
  /chase.mjpg  third-person chase view (car's built-in 'fpv' camera moved behind/above the car)
  /dash.mjpg   AV stack dashboard (6 RCCC cams + BEV + telemetry), read from --dash-file

Usage: .venv/bin/python stream_server.py [--host <tailscale-ip>] [--port 8090]
  (default host 127.0.0.1; or set AV_STREAM_HOST. Bind to a VPN/Tailscale address, not 0.0.0.0, to keep it private)
"""
import argparse, os, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import airsim
import cv2
import numpy as np

PAGE = b"""<!doctype html><html><head><meta name=viewport content="width=device-width,initial-scale=1">
<title>AirSim AV stream</title><style>body{margin:0;background:#111;color:#ddd;font:14px sans-serif}
h3{margin:8px 12px}img{width:100%;display:block}.w{max-width:1600px;margin:auto}</style></head>
<body><div class=w><h3>Chase view</h3><img src="/chase.mjpg"><h3>AV stack dashboard</h3><img src="/dash.mjpg"></div></body></html>"""


class Chase(threading.Thread):
    """Grabs the chase camera from AirSim in its own RPC connection."""
    def __init__(self, cam, fps):
        super().__init__(daemon=True)
        self.cam, self.dt, self.jpg, self.lock = cam, 1.0 / fps, None, threading.Lock()

    def run(self):
        c = None
        while True:
            try:
                if c is None:
                    c = airsim.CarClient()
                    c.confirmConnection()
                    c.simSetCameraPose(self.cam, airsim.Pose(airsim.Vector3r(-7.0, 0, -3.2),
                                                             airsim.to_quaternion(-0.22, 0, 0)))
                t = time.time()
                r = c.simGetImages([airsim.ImageRequest(self.cam, airsim.ImageType.Scene, False, False)])[0]
                if r.width:
                    img = np.frombuffer(r.image_data_uint8, np.uint8).reshape(r.height, r.width, 3)
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        with self.lock:
                            self.jpg = buf.tobytes()
                time.sleep(max(0.0, self.dt - (time.time() - t)))
            except Exception as e:
                print("chase:", e, flush=True)
                c = None
                time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("AV_STREAM_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--cam", default="fpv")
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--dash-file", default="/dev/shm/av_dash.jpg")
    args = ap.parse_args()

    chase = Chase(args.cam, args.fps)
    chase.start()

    def dash_frame():
        try:
            with open(args.dash_file, "rb") as f:
                return f.read()
        except OSError:
            return None

    def chase_frame():
        with chase.lock:
            return chase.jpg

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            src = {"/chase.mjpg": chase_frame, "/dash.mjpg": dash_frame}.get(self.path)
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(PAGE)
                return
            if src is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last = None
            try:
                while True:
                    jpg = src()
                    if jpg and jpg is not last and jpg != last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                        self.wfile.write(jpg + b"\r\n")
                        last = jpg
                    time.sleep(1.0 / args.fps)
            except (BrokenPipeError, ConnectionResetError):
                pass

    print(f"Streaming on http://{args.host}:{args.port}/", flush=True)
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
