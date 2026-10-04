"""What goes into the fly brain and what comes out, live, for the dashboard's brain view (/brain).

FlyNes.react() hands over each tick exactly what it gave FlyBrain.step() (the inject list:
neuron indices and voltage, per fly) and what step() returned (the neurons that fired, per fly
and brain step). Nothing here computes vision: it only names, sums and packs those values.
It works only while the brain view is open (watched() turns false 10 s after the last request).
"""
from __future__ import annotations

import base64
import time
import zlib
from collections import deque

import numpy as np

WATCH_SECONDS = 10.0
HISTORY = 100                           # ticks of raster kept (5 s)
VISUAL = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "LPLC2", "LC4", "LC10a", "LPLC1", "HSE", "HSN", "HSS", "H2")
DESCENDING = ("DNp01", "DNp02", "DNp04", "DNa02", "DNp15", "DNb03", "DNg111")
RASTER_PER_GROUP = 4


def pack(a: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(a).tobytes(), 6)).decode()


class BrainMonitor:
    def __init__(self, fly):
        from fly_nes import eye_layout
        self.fly = fly
        brain = fly.brain
        ct, side = brain.cell_type.astype(str), brain.side.astype(str)
        groups = {}
        for t in VISUAL + DESCENDING:
            for s in "LR":
                g = np.flatnonzero((ct == t) & (side == s))
                if len(g):
                    groups[f"{t} {s}"] = g
        self.group_names = list(groups)
        self.group_of = np.full(brain.n, len(groups), np.int32)        # neuron -> group (last = none)
        for k, g in enumerate(groups.values()):
            self.group_of[g] = k
        rng = np.random.default_rng(0)
        self.raster = [(nm, int(i)) for nm, g in groups.items()
                       for i in (g if len(g) <= RASTER_PER_GROUP else rng.choice(g, RASTER_PER_GROUP, replace=False))]
        self.raster_row = np.full(brain.n, -1, np.int32)
        for r, (_, i) in enumerate(self.raster):
            self.raster_row[i] = r
        lay = eye_layout()
        self.cells = lay["cells"]
        self.cell_pos = np.full(brain.n, -1, np.int32)                  # neuron -> place in the eye map
        self.cell_pos[self.cells] = np.arange(len(self.cells))
        self.layout = lay
        self.history: deque = deque(maxlen=HISTORY)
        self.last: dict | None = None
        self.seq = 0
        self.watched_until = 0.0

    # ---------------------------------------------------------------- the dashboard side
    def watch(self) -> None:
        self.watched_until = time.monotonic() + WATCH_SECONDS

    def watched(self) -> bool:
        return time.monotonic() < self.watched_until

    def meta(self) -> dict:
        lay = self.layout
        hue = (np.degrees(np.arctan2(lay["dir"][:, 1], lay["dir"][:, 0])) + 360) % 360
        return {"neurons": int(self.fly.brain.n), "cells": int(len(self.cells)),
                "x": np.round(lay["x"], 4).tolist(), "y": np.round(lay["y"], 4).tolist(),
                "hue": np.round(hue).astype(int).tolist(), "left": lay["left"].astype(int).tolist(),
                "raster": self.raster, "groups": self.group_names,
                "motion": getattr(self.fly, "motion_kind", "blobs"),
                "readouts": {r: self.fly.decoder.top.get(r, []) for r in self.fly.readouts}}

    def snapshot(self) -> dict | None:
        if self.last is None:
            return None
        rast = np.array(list(self.history), np.uint8) if self.history else np.zeros((0, 1), np.uint8)
        return {**self.last, "raster": pack(rast), "raster_ticks": len(self.history)}

    # ---------------------------------------------------------------- the fly side (every tick while watched)
    def _name(self, idx: np.ndarray) -> str:
        same = lambda cells: len(idx) == len(cells) and np.array_equal(idx, cells)
        motion = self.fly.motion.cells
        if isinstance(motion, dict):                                    # blob eyes: T4aL, T5aR...
            for nm, cells in motion.items():
                if same(cells):
                    return f"{nm[:-1]} {nm[-1]} (blobs)"
        elif same(motion):
            return "T4/T5 columns"
        for ch, d in self.fly.eyes.cells.items():                       # sprite channels: chase L, loom R...
            for s, cells in d.items():
                if same(cells):
                    return f"{ch} {s}"
        types = sorted(set(self.fly.brain.cell_type[idx].astype(str)))
        return ",".join(types[:3]) + (f" +{len(types) - 3}" if len(types) > 3 else "")

    def tick(self, inject, tick_flies, pad, pixels) -> None:
        B = self.fly.batch
        drive = np.zeros((len(self.cells), 2), np.float32)              # voltage into each T4/T5 cell, flies H and V
        inj = {}
        for idx, amt in inject:
            idx = np.asarray(idx)
            a = np.asarray(amt.get() if hasattr(amt, "get") else amt, np.float32)
            a = np.broadcast_to(a if a.ndim == 2 else a.reshape(1, -1) if a.ndim == 1 else a, (len(idx), B))
            inj[self._name(idx)] = [round(float(a[:, 0].mean()), 4), round(float(a[:, 1 % B].mean()), 4), int(len(idx))]
            pos = self.cell_pos[idx]
            hit = pos >= 0
            if hit.any():
                drive[pos[hit], 0] += a[hit, 0]
                drive[pos[hit], 1] += a[hit, 1 % B]
        ng = len(self.group_names)
        counts = np.zeros((ng + 1, 2), np.int64)
        bits = np.zeros(len(self.raster), bool)
        for flies in tick_flies:
            for f in (0, 1 % B):
                fired = np.asarray(flies[f])
                counts[:, f] += np.bincount(self.group_of[fired], minlength=ng + 1)
            rows = self.raster_row[np.asarray(flies[0])]
            bits[rows[rows >= 0]] = True
        self.history.append(np.packbits(bits))
        fly = self.fly
        p = fly.last_player
        self.seq += 1
        self.last = {
            "seq": self.seq, "game": fly.controls["game"], "seeking": bool(fly.seeking),
            "retina": pack(np.asarray(pixels, np.uint8)) if pixels is not None else None,
            "drive": pack(np.clip(drive * 400, 0, 255).astype(np.uint8)), "driven": int((drive[:, 0] > 0).sum()),
            "inj": inj, "spk": {nm: counts[k].tolist() for k, nm in enumerate(self.group_names)},
            "pad": [k for k, v in pad.items() if v], "acting": fly.decoder.state.get("acting", {}),
            "player": None if p is None else [round(p["x"]), round(p["y"])],
            "steps": len(tick_flies),
        }
