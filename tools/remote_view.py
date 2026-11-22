#!/usr/bin/env python3
"""
remote_view.py - see a script's camera window in a web browser on your computer.

WHY
  Your scripts draw their picture in an OpenCV window, which needs a monitor
  plugged into the Jetson. This launcher runs any of the scripts unchanged and
  sends that picture to a web page instead, so you can watch from your laptop
  over wifi. The keys the scripts use (q, c, SPACE, ENTER) work from the page,
  and so does the "drag a box around the target" step of find_target --enroll.

USAGE   (on the Jetson, over SSH; put the launcher in the same folder as the scripts)
    python3 remote_view.py find_person_v3.py
    python3 remote_view.py find_person_v3.py --specific
    python3 remote_view.py find_person_v3.py --enroll --name Alex
    python3 remote_view.py find_target_v2.py --enroll
    python3 remote_view.py obstacle_stop.py
    python3 remote_view.py obstacle_stop.py --live

  Everything after the script name is handed to the script exactly as you would
  type it. Options for the launcher itself go BEFORE the script name:
    python3 remote_view.py --port 9000 --fps 10 find_person_v3.py

  Then, on your computer, open the address it prints, e.g.  http://10.42.0.1:8080
  (find the Jetson's address yourself with:  hostname -I )

IN THE BROWSER
  * The page shows the live picture. Click the page, then use the same keys as
    the script describes, or use the buttons at the top:
        C         capture (find_target --enroll)
        SPACE     capture a face sample (find_person --enroll)
        ENTER     finish / save
        Q         quit the script (for obstacle_stop that also lands)
  * find_target --enroll: after you press C, the page freezes the picture.
    Drag a box around the target with the mouse, then press ENTER
    (ESC cancels and goes back to live view).

NOTES
  * Do NOT use the scripts' own --no-display flag with this launcher: it turns
    the picture off.
  * Anyone on the same wifi network who knows the address can watch the
    picture and press these keys, so use it on a network you trust.
  * Streaming costs a little CPU (JPEG encoding). If the Jetson is struggling,
    lower it:  --fps 8 --quality 55
  * Stop with Ctrl+C in the terminal, or the Q button / q key on the page.

Requires: python3 -m pip install opencv-python numpy   (already needed by the scripts)
"""

import argparse
import json
import os
import queue
import runpy
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

# --------------------------------------------------------------------- page
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drone view</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #111418; color: #d8dde3; font: 14px/1.4 system-ui, sans-serif; }
  #bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 8px 12px;
         background: #1b2027; border-bottom: 1px solid #2c333d; }
  #title { font-weight: 600; margin-right: 8px; }
  button { background: #2a3340; color: #e8edf2; border: 1px solid #3a4553; border-radius: 6px;
           padding: 5px 12px; font: inherit; cursor: pointer; }
  button:hover { background: #354154; }
  button.quit { border-color: #7a3a3a; }
  #hint { color: #e0b44c; margin-left: 8px; }
  #wrap { position: relative; display: inline-block; line-height: 0; }
  #v { display: block; max-width: 100vw; height: auto; }
  #c { position: absolute; left: 0; top: 0; width: 100%; height: 100%; cursor: crosshair; display: none; }
  #status { padding: 6px 12px; color: #8b95a1; font-size: 12px; }
</style></head>
<body>
<div id="bar">
  <span id="title">Connecting...</span>
  <button data-k="99"  title="C: capture (find_target --enroll)">C</button>
  <button data-k="32"  title="Space: capture a face sample (find_person --enroll)">Space</button>
  <button data-k="13"  title="Enter: finish / save">Enter</button>
  <button data-k="113" class="quit" title="Q: quit the script">Q (quit)</button>
  <span id="hint"></span>
</div>
<div id="wrap"><img id="v" alt="camera view"><canvas id="c"></canvas></div>
<div id="status"></div>
<script>
const img = document.getElementById('v'), cv = document.getElementById('c');
const titleEl = document.getElementById('title'), hint = document.getElementById('hint');
const statusEl = document.getElementById('status');
let mode = 'view', box = null, drag = null, gone = false, ignoreRoiUntil = 0;

function connect() { img.src = '/stream.mjpg?t=' + Date.now(); }
img.onerror = () => { if (!gone) { statusEl.textContent = 'Stream interrupted - retrying...'; setTimeout(connect, 1500); } };
img.onload = () => { statusEl.textContent = ''; };
connect();

function sendKey(code) { fetch('/key?code=' + code, { method: 'POST' }).catch(() => {}); }

// ---- drag-a-box mode (used by find_target --enroll) ----
function fit() { cv.width = img.clientWidth; cv.height = img.clientHeight; }
function redraw() {
  const g = cv.getContext('2d'); g.clearRect(0, 0, cv.width, cv.height);
  if (!box) return;
  g.lineWidth = 2; g.strokeStyle = '#3ddc84'; g.fillStyle = 'rgba(61,220,132,0.15)';
  g.fillRect(box.x, box.y, box.w, box.h); g.strokeRect(box.x, box.y, box.w, box.h);
}
function pos(e) { const r = cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
cv.addEventListener('mousedown', e => { drag = pos(e); box = { x: drag.x, y: drag.y, w: 0, h: 0 }; redraw(); });
cv.addEventListener('mousemove', e => {
  if (!drag) return; const p = pos(e);
  box = { x: Math.min(drag.x, p.x), y: Math.min(drag.y, p.y), w: Math.abs(p.x - drag.x), h: Math.abs(p.y - drag.y) }; redraw();
});
window.addEventListener('mouseup', () => { drag = null; });
function submitBox() {
  if (!box || box.w < 3 || box.h < 3) return;
  const s = img.naturalWidth / img.clientWidth;          // page pixels -> picture pixels
  const q = [box.x, box.y, box.w, box.h].map(v => Math.round(v * s));
  fetch('/roi?x=' + q[0] + '&y=' + q[1] + '&w=' + q[2] + '&h=' + q[3], { method: 'POST' });
  leaveRoi();
}
function cancelBox() { fetch('/roi?cancel=1', { method: 'POST' }); leaveRoi(); }
// Go back to normal keys right away, instead of waiting for the next status check.
function leaveRoi() { ignoreRoiUntil = Date.now() + 1000; setMode('view'); }

function setMode(m) {
  if (m === mode) return; mode = m;
  if (m === 'roi') { fit(); cv.style.display = 'block'; box = null; redraw();
                     hint.textContent = 'Drag a box around the target, then press Enter (Esc = cancel)'; }
  else { cv.style.display = 'none'; box = null; hint.textContent = ''; }
}
window.addEventListener('resize', () => { if (mode === 'roi') { fit(); redraw(); } });

// ---- keyboard ----
window.addEventListener('keydown', e => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (mode === 'roi') {
    if (e.key === 'Enter') { e.preventDefault(); submitBox(); }
    else if (e.key === 'Escape') { e.preventDefault(); cancelBox(); }
    return;
  }
  let code = null;
  if (e.key === 'Enter') code = 13; else if (e.key === ' ') code = 32; else if (e.key === 'Escape') code = 27;
  else if (e.key.length === 1) code = e.key.toLowerCase().charCodeAt(0);
  if (code !== null) { e.preventDefault(); sendKey(code); }
});
document.querySelectorAll('button[data-k]').forEach(b => b.addEventListener('click', () => {
  const k = +b.dataset.k;
  if (mode === 'roi') { if (k === 13) submitBox(); return; }
  sendKey(k);
}));

// ---- status polling ----
async function poll() {
  try {
    const s = await (await fetch('/state')).json();
    titleEl.textContent = s.title || 'Drone view';
    if (!(s.mode === 'roi' && Date.now() < ignoreRoiUntil)) setMode(s.mode);
    if (s.closed) { gone = true; statusEl.textContent = 'The script has finished.'; return; }
  } catch (e) { statusEl.textContent = 'Lost connection to the Jetson...'; }
  setTimeout(poll, 500);
}
poll();
</script></body></html>
"""


# ---------------------------------------------------------------------- hub
class Hub:
    """Shared state between the script (main thread) and the web server threads."""

    def __init__(self, max_fps: float, quality: int):
        self.cond = threading.Condition()
        self.jpeg = None
        self.seq = 0
        self.title = ""
        self.mode = "view"          # "view" or "roi"
        self.closed = False
        self.keys = queue.Queue(maxsize=64)
        self.roi_event = threading.Event()
        self.roi = (0, 0, 0, 0)
        self.min_gap = 1.0 / max(max_fps, 1.0)
        self.last_encode = 0.0
        self.quality = int(min(max(quality, 10), 95))
        self.clients = set()

    def publish(self, img, title=None, force=False):
        now = time.monotonic()
        if not force and now - self.last_encode < self.min_gap:
            return
        self.last_encode = now
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        with self.cond:
            self.jpeg = buf.tobytes()
            if title is not None:
                self.title = title
            self.seq += 1
            self.cond.notify_all()

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify_all()


def make_handler(hub: Hub):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):   # keep the terminal clean
            pass

        def _send(self, code, body=b"", ctype="text/plain; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _note_client(self):
            ip = self.client_address[0]
            if ip not in hub.clients:
                hub.clients.add(ip)
                print(f"[remote_view] browser connected from {ip}", flush=True)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/":
                    self._note_client()
                    self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/state":
                    self._send(200, json.dumps({"mode": hub.mode, "title": hub.title,
                                                "closed": hub.closed}).encode(), "application/json")
                elif path == "/snapshot.jpg":
                    if hub.jpeg is None:
                        self._send(503, b"no picture yet")
                    else:
                        self._send(200, hub.jpeg, "image/jpeg")
                elif path == "/stream.mjpg":
                    self._note_client()
                    self._stream()
                else:
                    self._send(404, b"not found")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            last = -1
            while True:
                with hub.cond:
                    hub.cond.wait_for(lambda: hub.seq != last or hub.closed, timeout=1.0)
                    jpeg, seq, closed = hub.jpeg, hub.seq, hub.closed
                if closed:
                    break
                if jpeg is None:
                    continue
                # a new picture, or a repeat every second so the connection stays alive
                last = seq
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                self.wfile.flush()

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/key":
                    code = int(q.get("code", ["-1"])[0])
                    try:
                        hub.keys.put_nowait(code)
                    except queue.Full:
                        pass
                    self._send(204)
                elif u.path == "/roi":
                    if hub.mode != "roi":
                        self._send(409, b"not selecting a box right now")
                        return
                    if q.get("cancel"):
                        hub.roi = (0, 0, 0, 0)
                    else:
                        x, y, w, h = (int(float(q[k][0])) for k in ("x", "y", "w", "h"))
                        hub.roi = (x, y, w, h)
                    hub.roi_event.set()
                    self._send(204)
                else:
                    self._send(404, b"not found")
            except (ValueError, KeyError):
                self._send(400, b"bad request")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    return Handler


# --------------------------------------------------- stand-ins for OpenCV windows
def install_cv2_patches(hub: Hub):
    """Replace the few cv2 window functions the scripts use with web-page versions."""

    def imshow(winname, mat):
        hub.publish(mat, title=winname)

    def waitKey(delay=0):
        try:
            if delay and delay > 0:
                return hub.keys.get(timeout=delay / 1000.0)
            return hub.keys.get()          # delay 0 = wait for a key
        except queue.Empty:
            return -1

    def selectROI(windowName, img, showCrosshair=True, fromCenter=False, printNotice=True):
        hub.roi_event.clear()
        hub.roi = (0, 0, 0, 0)
        hub.publish(img, title=windowName, force=True)
        hub.mode = "roi"
        print("[remote_view] waiting for you to drag a box in the browser, then press Enter...",
              flush=True)
        try:
            while not hub.roi_event.wait(0.2):   # short waits so Ctrl+C still works
                pass
        finally:
            hub.mode = "view"
        return hub.roi

    def destroyWindow(winname):
        hub.mode = "view"

    def destroyAllWindows():
        hub.mode = "view"

    def noop(*a, **k):
        return None

    cv2.imshow = imshow
    cv2.waitKey = waitKey
    cv2.selectROI = selectROI
    cv2.destroyWindow = destroyWindow
    cv2.destroyAllWindows = destroyAllWindows
    for name in ("namedWindow", "moveWindow", "resizeWindow", "setWindowTitle"):
        setattr(cv2, name, noop)


def jetson_addresses():
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=3).stdout.split()
        ips = [a for a in out if "." in a]
        if ips:
            return ips
    except Exception:
        pass
    return []


def main():
    argv = sys.argv[1:]
    idx = next((i for i, a in enumerate(argv) if a.endswith(".py")), None)

    ap = argparse.ArgumentParser(
        prog="remote_view.py",
        usage="remote_view.py [--port N] [--bind ADDR] [--fps N] [--quality N] SCRIPT.py [script options]",
        description="Run one of the camera scripts and stream its window to a web browser.")
    ap.add_argument("--port", type=int, default=8080, help="web port (default 8080)")
    ap.add_argument("--bind", default="0.0.0.0",
                    help="address to listen on (default 0.0.0.0 = reachable from your computer)")
    ap.add_argument("--fps", type=float, default=15, help="most pictures per second to send (default 15)")
    ap.add_argument("--quality", type=int, default=70, help="JPEG quality 10-95 (default 70)")
    if idx is None:
        ap.parse_args(argv)   # prints help or a clear error for bad launcher options
        ap.error("give the script to run, e.g.  remote_view.py find_person_v3.py --specific")
    args = ap.parse_args(argv[:idx])
    script, script_args = argv[idx], argv[idx + 1:]
    if not os.path.isfile(script):
        sys.exit(f"Can't find the script '{script}'. Run this from the folder that holds it "
                 f"(you are in {os.getcwd()}).")
    if "--no-display" in script_args:
        print("[remote_view] note: '--no-display' turns the picture off, so there will be nothing to see.",
              flush=True)

    hub = Hub(args.fps, args.quality)
    try:
        server = ThreadingHTTPServer((args.bind, args.port), make_handler(hub))
    except OSError as e:
        sys.exit(f"Can't open port {args.port}: {e}\n"
                 f"Is another copy still running? Try a different port:  --port {args.port + 1}")
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()

    ips = jetson_addresses()
    print("\n[remote_view] On your computer, open one of:", flush=True)
    for ip in (ips or ["<jetson-address>"]):
        print(f"                http://{ip}:{args.port}", flush=True)
    print("[remote_view] (find the address yourself with:  hostname -I )\n", flush=True)

    install_cv2_patches(hub)
    os.environ.setdefault("DISPLAY", ":remote")    # the scripts only draw a window if DISPLAY is set
    sys.argv = [script] + script_args
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    try:
        runpy.run_path(script, run_name="__main__")
    except KeyboardInterrupt:
        print("\n[remote_view] Ctrl+C - exiting", flush=True)
    finally:
        hub.close()
        time.sleep(1.5)     # stay up briefly so the page can show "The script has finished."


if __name__ == "__main__":
    main()