"""Live dashboard: EmulatorJS NES on the left, fly connectome on the right.

The page runs the emulator, reads the NES sprite table every tick, POSTs it to
/frame and applies the pad that comes back. Only one page drives the fly at a
time: another tab gets 409 until the driver's game has stopped advancing (a closed,
paused or background tab) for OWNER_TIMEOUT.
"""
from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import threading
import time
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

MAX_SPIKES = 6000
RETINA = (120, 128)                 # the page's eye image: rows, cols (fly_nes.RETINA)
PAD_KEYS = ("a", "b", "select", "start", "up", "down", "left", "right")
OWNER_TIMEOUT = 2.0
HUMAN_TIMEOUT = 0.6                   # s: buttons held on the dashboard pad let go unless refreshed
ARROWS = {"H": "◀ ▶", "V": "▲ ▼"}
HERE = Path(__file__).resolve().parent
EJS_CANDIDATES = (
    HERE / "node_modules" / "@emulatorjs" / "emulatorjs" / "data",
    HERE / "node_modules" / "@emulatorjs" / "emulatorjs",
)
# Your own games: the dashboard lists the ROMs in here (no ROMs ship with this repo)
GAME_DIR = Path(os.environ.get("FLY_ROMS") or HERE.parent / "roms")
ROM_TYPES = {".nes", ".zip", ".7z", ".fds", ".unf"}


def nes_inside(path: Path) -> bool:
    """A zip must hold an NES game (not e.g. the Game Boy Advance "Classic NES Series" ports)."""
    if path.suffix.lower() != ".zip":
        return True
    try:
        with zipfile.ZipFile(path) as z:
            return any(Path(n).suffix.lower() in ROM_TYPES for n in z.namelist())
    except zipfile.BadZipFile:
        return False
MAX_ROM = 4 << 20                    # bytes: the biggest NES games are 1 MB


def check_rom(name: str, data: bytes) -> str | None:
    """Why an uploaded file can't be added (None = it's an NES game the server can run)."""
    ext = Path(name).suffix.lower()
    if ext not in (".nes", ".zip"):
        return "only .nes files, or .zip files holding one"
    if not data or len(data) > MAX_ROM:
        return "file is empty or too big for an NES game"
    if ext == ".nes":
        return None if data[:4] == b"NES" else "not an NES ROM (no iNES header)"
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            inner = [n for n in z.namelist() if n.lower().endswith(".nes")]
            if not inner:
                return "the zip holds no .nes file"
            with z.open(inner[0]) as f:
                return None if f.read(4) == b"NES" else "the .nes in the zip is not an NES ROM"
    except zipfile.BadZipFile:
        return "not a valid zip file"


MIME = {
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".css": "text/css",
    ".wasm": "application/wasm",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".html": "text/html; charset=utf-8",
    ".data": "application/octet-stream",
    ".zip": "application/zip",
}


def ejs_root() -> Path | None:
    for path in EJS_CANDIDATES:
        if (path / "loader.js").is_file():
            return path
    return None


def _num(v):
    return round(float(v), 2)


class Dashboard:
    def __init__(self, fly, port: int = 8777, rom: Path | None = None, rom_dir: Path | None = None,
                 engine: str = "browser"):
        self.fly = fly
        self.engine = engine
        self.emu = None                      # ServerEmulator when the game runs in this process
        self._monitor = None                 # BrainMonitor for /brain, made on first use
        self.rom = rom                       # None until a game is added
        self.rom_dir = rom_dir
        self.port = port
        self.lock = threading.Lock()
        self.human: dict = {"buttons": {}, "drive": False}   # the dashboard pad (a person's buttons)
        self.human_seen = 0.0
        brain = fly.brain
        if brain.positions is None:
            raise SystemExit("brain.npz has no positions; run `flybrain build`")
        ok = ~np.isnan(brain.positions).any(axis=1)
        self.pos_index = np.full(brain.n, -1, np.int32)
        self.pos_index[ok] = np.arange(ok.sum(), dtype=np.int32)
        xy = brain.positions[ok][:, [0, 2]].copy()
        side = brain.side[ok]
        if np.nanmean(xy[side == "R", 0]) < np.nanmean(xy[side == "L", 0]):
            xy[:, 0] = -xy[:, 0]
        lo, hi = np.percentile(xy, 0.2, axis=0), np.percentile(xy, 99.8, axis=0)
        scale = (hi - lo).max()
        pad = (scale - (hi - lo)) / 2
        norm = np.clip((xy - lo + pad) / scale, 0, 1) * 1000
        mapped = lambda idx: [int(i) for i in self.pos_index[idx] if i >= 0]  # noqa: E731
        groups = {}
        dec = fly.decoder
        self.labels = {}
        for name in dec.w:                                   # neurons each population readout listens to
            for axis, arrow in ARROWS.items():
                key = f"{name}{axis}"
                self.labels[key] = (f"{name} {arrow}", " ".join(list(dict.fromkeys(n[:-1] for n in dec.top.get(name, [])))[:2]))
                groups[f"{key}_L"] = mapped(dec.dn[dec.w[name] > 0])
                groups[f"{key}_R"] = mapped(dec.dn[dec.w[name] < 0])
        for channel, per_side in fly.features.cells.items():
            for s, idx in per_side.items():
                groups[f"{channel}{s}"] = mapped(idx)
        self.static = json.dumps({
            "x": norm[:, 0].round().astype(int).tolist(),
            "y": norm[:, 1].round().astype(int).tolist(),
            "groups": groups,
            "labels": self.labels,
            "readout_neurons": {n: int((w != 0).sum()) for n, w in dec.w.items()},
            "neurons": int(brain.n),
            "connections": int(len(brain.indices)),
            "mapped": int(ok.sum()),
            "device": brain.device,
            "flies": int(fly.batch),
            "ejs": ejs_root() is not None,
            "local_rom": bool(self.rom and self.rom.is_file()),
            "engine": engine,
        })
        self.cond = threading.Condition()
        self.payload: str | None = None
        self.seq = 0
        self.rng = np.random.default_rng(0)
        self.owner: str | None = None
        self.owner_seen = 0.0                # last time the driving tab's game moved on
        self.owner_frame = -1

    def reward_state(self) -> dict:
        """The reward feed for /rewards: event catalogue, this game's settings, the last minute."""
        import reward_feed as rf
        fly = self.fly
        feed = fly.reward if isinstance(fly.reward, rf.RewardFeed) else None
        out = {"game": fly.controls["game"], "genre": fly.controls.get("genre", ""), "rom": fly.rom_name,
               "events": rf.EVENTS, "stimulus": rf.STIMULUS, "stim_max": rf.STIM_MAX,
               "learning": fly.learner.status() if fly.learner else None,
               "searching": fly.finder is not None, "seeking": bool(fly.seeking)}
        if feed is not None:
            out.update(feed.state())
        else:
            cfg = rf.load_config(fly.rom_name, out["genre"])
            out["config"] = {**cfg, "ram_rules": [{**r, "addr": f"{r['addr']:#05x}"} for r in cfg["ram_rules"]]}
        return out

    def set_reward_config(self, body: dict) -> dict:
        """Save this game's reward settings and apply them to the running fly at once."""
        import reward_feed as rf
        fly = self.fly
        genre = fly.controls.get("genre", "")
        cfg = rf.default_config(genre) if body.get("reset") else rf.load_config(fly.rom_name, genre)
        if not body.get("reset"):
            for k, v in (body.get("weights") or {}).items():
                if k in rf.EVENTS:
                    cfg["weights"][k] = max(-20.0, min(20.0, float(v)))
            if "taste" in body:
                cfg["taste"] = bool(body["taste"])
            if "ram_rules" in body:
                cfg["ram_rules"] = rf.clean_rules(body["ram_rules"])
            if "stimulus" in body:
                cfg["stimulus"].update(rf.clean_stimulus(body["stimulus"]))
        rf.save_config(fly.rom_name, cfg)
        with self.lock:
            if isinstance(fly.reward, rf.RewardFeed):
                fly.reward.set_config(cfg)
            fly.stim_gain = dict(cfg["stimulus"])
        return self.reward_state()

    def brain_monitor(self):
        """The brain view's recorder (brain_monitor.py), made on first use and handed to the fly."""
        if self._monitor is None:
            from brain_monitor import BrainMonitor
            mon = BrainMonitor(self.fly)
            mon.meta_json = json.dumps(mon.meta()).encode()
            self._monitor = mon
            self.fly.monitor = mon
        return self._monitor

    def library(self) -> dict[str, Path]:
        """ROM file name -> path, for the dashboard's game list."""
        if self.rom_dir is None or not self.rom_dir.is_dir():
            return {self.rom.name: self.rom} if self.rom and self.rom.is_file() else {}
        roms = {p.name: p for p in sorted(self.rom_dir.iterdir())
                if p.suffix.lower() in ROM_TYPES and p.is_file() and p.stat().st_size < 16 << 20 and nes_inside(p)}
        if self.rom and self.rom.is_file():
            roms.setdefault(self.rom.name, self.rom)
        return roms

    def add_rom(self, name: str, data: bytes) -> dict:
        """Upload from the Play page: copy the player's own ROM into the game folder."""
        name = Path(name).name.strip()
        why = check_rom(name, data)
        if why:
            return {"ok": False, "error": why}
        folder = self.rom_dir if self.rom_dir is not None else GAME_DIR
        folder.mkdir(parents=True, exist_ok=True)
        target, k = folder / name, 2
        while target.exists() and target.read_bytes() != data:       # never overwrite a different game
            target = folder / f"{Path(name).stem} ({k}){Path(name).suffix}"
            k += 1
        if not target.exists():
            target.write_bytes(data)
        return {"ok": True, "name": target.name, "library": list(self.library())}

    def switch_game(self, name: str) -> dict:
        """The page loaded another game (from the list, or a file of its own)."""
        lib = self.library()
        if name in lib:
            self.rom = lib[name]
            if self.emu is not None:
                self.emu.switch(self.rom)                    # the server's own emulator loads it too
            elif self.engine == "server":                    # started with no game: the first one starts it
                from server_emu import ServerEmulator
                self.emu = ServerEmulator(self, self.rom)
        with self.lock:
            c = self.fly.switch_game(Path(name).stem)
        return {"game": c["game"], "genre": c["genre"], "ok": c["ok"], "rom": name}

    @property
    def connected(self) -> bool:
        return self.owner is not None and time.monotonic() - self.owner_seen < OWNER_TIMEOUT

    def handle_frame(self, sprites: list[dict], frame: int, client: str, tall: bool = False,
                     pixels: np.ndarray | None = None, ram: np.ndarray | None = None,
                     objects: list[dict] | None = None) -> dict[str, bool] | None:
        """Run one tick for `client`; None if another client is driving."""
        with self.lock:
            now = time.monotonic()
            if client != self.owner and self.connected:
                return None
            if client != self.owner or frame != self.owner_frame:   # a paused / background tab's game
                self.owner, self.owner_seen, self.owner_frame = client, now, frame   # stops, and lets go
            pad = self.fly.react(sprites, frame, tall, pixels, ram, objects=objects, human=self.human_now())
        self.publish(pad)
        return pad

    def set_human(self, body: dict) -> dict:
        """POST /pad: the buttons a person holds now (sent again while held) and whether they drive alone."""
        held = body.get("buttons") or {}
        with self.lock:
            self.human = {"buttons": {k: bool(held.get(k)) for k in PAD_KEYS},
                          "drive": bool(body.get("drive"))}
            self.human_seen = time.monotonic()
        return self.human

    def human_now(self) -> dict | None:
        """The person's pad while fresh: a lost tab or connection never leaves a button held."""
        h = self.human
        if time.monotonic() - self.human_seen > HUMAN_TIMEOUT:
            return None
        return h if h["drive"] or any(h["buttons"].values()) else None

    def publish(self, pad: dict[str, bool]) -> None:
        fly = self.fly
        spikes = self.pos_index[fly.frame_spikes]
        spikes = spikes[spikes >= 0]
        if len(spikes) > MAX_SPIKES:
            spikes = self.rng.choice(spikes, MAX_SPIKES, replace=False)
        counts = fly.decoder.snapshot()
        now = {k: v >= 1.0 for k, v in counts.items()}
        box = lambda o: {k: _num(o[k]) for k in ("x", "y", "w", "h")}  # noqa: E731
        st = fly.decoder.state
        payload = {
            "connected": self.connected,
            "frame": int(fly.last_frame),
            "you": None if fly.last_player is None else box(fly.last_player),
            "sprites": len(fly.last_sprites),
            "threats": [box(t) for t in fly.last_others],
            "wander": fly.wander is not None,
            "cmd": pad,
            "human": self.human_now(),
            "spikes": spikes.tolist(),
            "total": int(len(fly.frame_spikes)),
            "counts": {k: _num(v) for k, v in counts.items()},
            "now": now,
            "drive": {k: _num(v) for k, v in fly.eyes.display().items()},
            "motion": {k: _num(v) for k, v in fly.motion.last.items() if v},
            "vision": fly.vision,
            "learning": fly.learner.status() if fly.learner is not None else None,
            "finding": fly.finder is not None,
            "game": {k: fly.controls[k] for k in ("game", "genre", "ok")},
            "state": {"facing": bool(st.get("facing")), "fleeing": bool(st.get("fleeing")),
                      "seeking": bool(fly.seeking), "acting": st.get("acting", {})},
            "ms": round(fly.step_ms, 1),
            "engine": self.engine,
            "fps": round(self.emu.fps, 1) if self.emu is not None else None,
        }
        with self.cond:
            self.payload = json.dumps(payload)
            self.seq += 1
            self.cond.notify_all()

    def start(self, open_browser: bool = True) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", self.port), _handler(self))
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        threading.Thread(target=self._idle_loop, daemon=True).start()
        url = f"http://127.0.0.1:{self.port}/"
        print(f"dashboard: {url}", flush=True)
        if open_browser:
            webbrowser.open(url)

    def _idle_loop(self) -> None:
        """Keep the brain view alive while no emulator is attached."""
        pad = {k: False for k in ("a", "b", "select", "start", "up", "down", "left", "right")}
        while True:
            time.sleep(0.2)
            if self.connected:
                continue
            with self.lock:
                self.fly.idle_tick()
            self.publish(pad)


def _handler(dash: Dashboard):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body: bytes, ctype: str, cache: str = "no-store", status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def _rom_path(self) -> Path | None:
            """The current game (/local-rom), or one from the library (/rom?name=...)."""
            url = urlparse(self.path)
            path = unquote(url.path)
            if path == "/local-rom":
                return dash.rom if dash.rom and dash.rom.is_file() else None
            if path == "/rom":
                name = parse_qs(url.query).get("name", [""])[0]
                return dash.library().get(name)                # only files the list offers
            return None

        def do_HEAD(self):
            """EmulatorJS asks for the ROM's size before downloading it."""
            rom = self._rom_path()
            if rom is None:
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(rom.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def do_POST(self):
            route = urlparse(self.path).path
            if route not in ("/frame", "/game", "/rewards/config", "/pad", "/rom/upload"):
                return self.send_error(404)
            length = int(self.headers.get("Content-Length") or 0)
            if route == "/rom/upload":                       # raw file bytes; the name in ?name=
                if length > MAX_ROM:
                    return self._send(b'{"ok":false,"error":"file too big for an NES game"}', "application/json", status=413)
                name = parse_qs(urlparse(self.path).query).get("name", [""])[0]
                res = dash.add_rom(name, self.rfile.read(length))
                return self._send(json.dumps(res).encode(), "application/json", status=200 if res["ok"] else 400)
            try:
                body = json.loads(self.rfile.read(length) if length else b"{}")
            except json.JSONDecodeError:
                return self.send_error(400)
            if route == "/frame" and dash.engine == "server":
                return self._send(b'{"busy":true}', "application/json", status=409)   # the server plays
            if route == "/pad":
                return self._send(json.dumps(dash.set_human(body)).encode(), "application/json")
            if route == "/rewards/config":
                return self._send(json.dumps(dash.set_reward_config(body)).encode(), "application/json")
            if route == "/game":
                name = Path(str(body.get("name") or "")).name
                if not name:
                    return self.send_error(400)
                return self._send(json.dumps(dash.switch_game(name)).encode(), "application/json")
            pixels = None
            if body.get("pixels"):
                raw = np.frombuffer(base64.b64decode(body["pixels"]), np.uint8)
                if raw.size == RETINA[0] * RETINA[1]:
                    pixels = raw.reshape(RETINA)
            ram = None
            if body.get("ram"):
                ram = np.frombuffer(base64.b64decode(body["ram"]), np.uint8)
            pad = dash.handle_frame(body.get("sprites") or [], int(body.get("frame") or 0),
                                    str(body.get("client") or "anon"), bool(body.get("tall")), pixels, ram)
            if pad is None:
                return self._send(b'{"busy":true}', "application/json", status=409)
            return self._send(json.dumps(pad).encode(), "application/json")

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == "/":
                html = (HERE / "dashboard.html").read_text(encoding="utf-8")
                return self._send(html.encode(), "text/html; charset=utf-8")
            if path == "/rewards":
                html = (HERE / "rewards.html").read_text(encoding="utf-8")
                return self._send(html.encode(), "text/html; charset=utf-8")
            if path == "/rewards/state.json":
                return self._send(json.dumps(dash.reward_state()).encode(), "application/json")
            if path == "/retro.css":                          # the house style every page shares
                return self._send((HERE / "retro.css").read_bytes(), "text/css; charset=utf-8")
            if path == "/brain":
                html = (HERE / "brain.html").read_text(encoding="utf-8")
                return self._send(html.encode(), "text/html; charset=utf-8")
            if path in ("/brain/meta.json", "/brain/tick.json"):
                mon = dash.brain_monitor()
                if path == "/brain/meta.json":
                    return self._send(mon.meta_json, "application/json")
                mon.watch()                                     # keeps the fly recording while this page polls
                snap = mon.snapshot()
                return self._send(json.dumps(snap or {"waiting": True}).encode(), "application/json")
            if path == "/screen.png":
                if dash.emu is None:
                    return self.send_error(404)
                return self._send(dash.emu.screen_png, "image/png")
            if path == "/static.json":
                static = json.loads(dash.static)
                static.update(rom_file=dash.rom.name if dash.rom else "", rom_name=dash.rom.stem if dash.rom else "",
                              library=list(dash.library()))
                return self._send(json.dumps(static).encode(), "application/json")
            if path in ("/local-rom", "/rom"):
                rom = self._rom_path()
                if rom is None:
                    return self.send_error(404)
                return self._send(rom.read_bytes(), "application/octet-stream")
            if path.startswith("/ejs/"):
                root = ejs_root()
                if root is None:
                    return self.send_error(404)
                rel = Path(path[len("/ejs/"):])
                if ".." in rel.parts:
                    return self.send_error(403)
                target = (root / rel).resolve()
                if not str(target).startswith(str(root.resolve())) or not target.is_file():
                    return self.send_error(404)
                ctype = MIME.get(target.suffix.lower()) or mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                return self._send(target.read_bytes(), ctype, cache="public, max-age=86400")
            if path != "/events":
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with dash.cond:
                        fresh = dash.cond.wait_for(lambda: dash.seq != seen, timeout=15)
                        seen, data = dash.seq, dash.payload
                    self.wfile.write(f"data: {data}\n\n".encode() if fresh and data else b": ping\n\n")
                    self.wfile.flush()
            except OSError:
                return

    return Handler
