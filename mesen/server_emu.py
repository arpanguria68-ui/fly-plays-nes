"""The game runs inside the fly server (cynes NES emulator), not in a browser tab.

Browsers pause games in background tabs, so a game in the page stops whenever you
look elsewhere. Here the game keeps running at real speed whatever the browser
does; the dashboard only shows it. Each tick (3 NES frames, like the page did) the
fly gets the sprites, the picture and the RAM, and its pad goes to controller 1.
"""
from __future__ import annotations

import threading
from collections import deque
import time
import zipfile
import zlib
import struct
from pathlib import Path

import numpy as np
from scipy import ndimage

import cynes
from cynes import NES

TICK = 3                                     # NES frames per fly tick (20 ticks a second)
OAM_VOTES = 100                              # ticks (5 s) over which $0200 is judged to be the sprite table
OAM_SHARE = 0.3                              # ...and the share of them it must match the picture
FPS = 60.0988                                # NTSC NES
BITS = {"a": cynes.NES_INPUT_A, "b": cynes.NES_INPUT_B, "select": cynes.NES_INPUT_SELECT,
        "start": cynes.NES_INPUT_START, "up": cynes.NES_INPUT_UP, "down": cynes.NES_INPUT_DOWN,
        "left": cynes.NES_INPUT_LEFT, "right": cynes.NES_INPUT_RIGHT}


def rom_bytes(path: Path) -> bytes:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".nes"))
            return z.read(name)
    return path.read_bytes()


def shadow_oam(nes) -> list[dict]:
    """Most NES games keep a copy of the sprite table at $0200 and DMA it to the PPU."""
    out = []
    for i in range(64):
        y = nes[0x200 + i * 4]
        if y < 0xEF:
            out.append({"i": i, "x": nes[0x203 + i * 4], "y": y, "tile": nes[0x201 + i * 4], "attr": nes[0x202 + i * 4]})
    return out


def drawn(spr: list[dict], frame: np.ndarray) -> bool:
    """Is this really the sprite table? Most of its sprites must be visible on the screen
    where it says (Galaga keeps other data at $0200)."""
    if len(spr) < 2:
        return False
    bg = np.median(frame.reshape(-1, 3), axis=0)
    hits = 0
    for s in spr[:24]:
        box = frame[s["y"] + 1:s["y"] + 9, s["x"]:s["x"] + 8].astype(int)
        hits += box.size > 0 and (np.abs(box - bg).sum(axis=2) > 60).any()
    return hits >= 0.5 * min(len(spr), 24)          # half: sprites can be blank tiles (small Mario: 4 of 8)


def pixel_objects(frame: np.ndarray) -> list[dict]:
    """Fallback for games without a $0200 sprite copy: bright patches on the screen."""
    g = frame.max(axis=2) > 60
    lab, _ = ndimage.label(ndimage.binary_dilation(g, iterations=1))
    out = []
    for sl in ndimage.find_objects(lab):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if 4 <= max(w, h) <= 40 and w * h >= 12:
            out.append({"x": (sl[1].start + sl[1].stop) / 2, "y": (sl[0].start + sl[0].stop) / 2,
                        "w": float(w), "h": float(h), "n": 1})
    return out


def retina(frame: np.ndarray) -> np.ndarray:
    return frame.mean(axis=2).reshape(120, 2, 128, 2).mean(axis=(1, 3)).astype(np.uint8)


def png(frame: np.ndarray) -> bytes:
    h, w, _ = frame.shape
    raw = b"".join(b"\x00" + frame[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


class ServerEmulator:
    def __init__(self, dash, rom: Path):
        self.dash = dash
        self.lock = threading.Lock()
        self.rom = rom
        self.nes: NES | None = None
        self.frame = np.zeros((240, 256, 3), np.uint8)
        self.frame_no = 0
        self.screen_png = png(self.frame)
        self.screen_seq = 0
        self.fps = 0.0
        self.oam_votes: deque = deque(maxlen=OAM_VOTES)
        self._load(rom)
        threading.Thread(target=self._run, daemon=True).start()

    def _load(self, rom: Path) -> None:
        tmp = Path(__file__).resolve().parent / "checkpoints" / "_rom.nes"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(rom_bytes(rom))                  # cynes loads from a file
        self.nes = NES(str(tmp))
        self.rom = rom
        self.frame_no = 0
        self.oam_votes.clear()

    def switch(self, rom: Path) -> None:
        with self.lock:
            self._load(rom)

    def _run(self) -> None:
        period = TICK / FPS
        ram = np.zeros(0x800, np.uint8)
        next_t = time.perf_counter()
        ticks, t_fps = 0, time.perf_counter()
        while True:
            with self.lock:
                nes = self.nes
                spr = shadow_oam(nes)
                # judged over seconds, not per tick: with few sprites on screen one tick can fail
                # (small Mario is 8 tiles, 4 of them blank) and the pixel fallback sees nothing
                # on a sky-blue screen
                self.oam_votes.append(drawn(spr, self.frame))
                own_oam = np.mean(self.oam_votes) >= OAM_SHARE
                ram[:] = [nes[a] for a in range(0x800)]
                objects = None if own_oam else pixel_objects(self.frame)
                pad = self.dash.handle_frame(spr, self.frame_no + 1, "server", False, retina(self.frame), ram,
                                             objects=objects)
                nes.controller = sum(BITS[k] for k, v in (pad or {}).items() if v)
                self.frame = nes.step(TICK)
                self.frame_no += TICK
            ticks += 1
            self.screen_png = png(self.frame)              # a picture every tick (20 a second) for the dashboard
            self.screen_seq += 1
            now = time.perf_counter()
            if now - t_fps >= 1.0:
                self.fps, ticks, t_fps = ticks * TICK / (now - t_fps), 0, now
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()              # the brain is slower than real time: don't pile up
