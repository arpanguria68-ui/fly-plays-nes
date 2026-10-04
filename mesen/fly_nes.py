"""Drive a NES game with the frozen fly connectome.

EmulatorJS runs the game in the dashboard page and sends, each tick, the NES
sprite table (OAM) and the picture (128x120 grey). Two senses, --vision both|eyes|sprites:

* sprites: objects from the sprite table drive the fly's own object detectors
  (LC10a chase, LPLC2 looming, LC4 threat, LPLC1 small objects).
* eyes: the picture drives the elementary motion detectors T4/T5 (ON/OFF) when
  something grows on the retina; the connectome carries it on to LPLC2 and the
  escape neurons. Photoreceptors and the first relays are skipped: they use graded
  voltages a spiking model can't carry (an image on the photoreceptors dies there).
  --motion columns instead drives every T4/T5 cell by its own column and preferred
  direction (both read off the wiring, flybrain.columns). Measured: the connectome then
  turns wide-field motion into direction-selective HS/H2 responses and ~15-20 side-selective
  descending neurons (a "flow" readout the learner sees); shuffling the eye map abolishes
  both. Looming reaches LPLC2 on the correct side but not the descending neurons.

The brain steps, and its descending neurons decide an 8-button NES pad.

Two axes, two flies. A fly only sees left vs right, but NES games move in four
directions. So the brain runs as a batch: fly H sees the screen as it is
(left/right), fly V sees it turned 90 degrees (up = its left, down = its right).
Same wiring, own noise. With --voters N each axis gets N flies that vote.

Outputs: the whole descending population (1,314 neurons, the brain's cables to the
body), not a few hand-picked cells. At startup each visual channel is shown on the
left and on the right; the descending neurons that answer differently become that
readout (flee: DNp04/DNp02/DNp01..., dodge: DNp03/DNp35..., chase: DNg111/DNae002/
DNa02...). In play, whichever readout leans past its threshold moves the player on
that fly's axis: toward (chase) or away (flee, dodge).

    .\\.venv\\Scripts\\python.exe mesen\\connect.py          # dashboard + EmulatorJS
    # open http://127.0.0.1:8777/ ; the page loads the ROM and the fly plays
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from flybrain import FlyBrain
from flybrain.brain import cuda_available
from flybrain.eyes import CHANNELS, ENCODER
from fly_dashboard import GAME_DIR, Dashboard, nes_inside
from games import controls_for
from learner import Learner, RewardFinder, heads_for, profile_for
from reward_feed import RewardFeed, load_config as load_reward_config

AXES = ("H", "V")                   # fly b looks along AXES[b % 2]
SIDES = {"H": ("left", "right"), "V": ("up", "down")}   # pad keys for the fly's L / R side
OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}
# Population readouts: which way the brain's descending neurons (all 1,314 of them) lean.
# Each is calibrated at startup by stimulating its visual channel on the left vs the right.
#   name: (channels driven during calibration, window in brain steps, what the pad does)
READOUTS = {
    "flee": (("threat", "loom"), 5, "away"),    # LC4 / LPLC2 -> DNp04, DNp02, DNp01 (giant fibre)...
    "looming": (("motion",), 5, "away"),        # eyes: outward motion on T4/T5 -> LPLC2 -> DNp01/DNp04...
    "dodge": (("shot",), 8, "away"),            # LPLC1 -> DNp03, DNp35...: small approaching objects
    "chase": (("chase",), 25, "toward"),        # LC10a -> DNg111, DNae002, DNa02...: target pursuit
}
PRIORITY = ("flee", "looming", "dodge", "chase")
# With --motion columns the eyes also carry wide-field optic flow (the world sliding left or right).
# The real wiring passes it to ~60-140 descending neurons, mirror-symmetric (DNb03, DNp15, DNa07,
# DNg41...; 3 with the eye map shuffled). It is not a threat or a target, so it presses nothing:
# it is one more readout the learner sees ("the screen is scrolling left").
FLOW_READOUT = {"flow": (("flow",), 5, "none")}
CAL_AMOUNT = 0.6                    # voltage per step on the stimulated channel during calibration
CAL_REPS = 4                        # left and right presentations per readout
CAL_SETTLE = 15                     # brain steps of silence before each presentation
TOP_NEURONS = 40                    # descending neurons kept per readout (most side-selective)
THRESHOLD_SD = 3.0                  # act when a readout leans this many noise SDs to one side
MIN_THRESHOLD = {"flee": 0.5, "looming": 0.5, "dodge": 0.3, "chase": 0.3, "flow": 0.3}   # ...and at least this far (1.0 = calibration stimulus);
#   flee needs half a close attack, so a big but distant sprite (size-looming) does not scare it
BASELINE_TAU = 20.0                 # seconds; slow running mean of every neuron's resting rate
ATTACK_RANGE = 36.0                 # px between centres: the fly may strike when it turns toward a target this close
COOLDOWN = {"attack": 0.6}
HOLD_TICKS = {"attack": 2}
CLUSTER_PX = 9.0                    # sprites this close belong to one object (NES metasprites are 8 px tiles)
MATCH_PX = 14.0                     # an object moves at most this far between ticks
SCREEN_W, SCREEN_H = 256.0, 240.0
PLAY_TOP, PLAY_BOTTOM = 8.0, 231.0         # sprite y outside this is hidden (y >= 239) or HUD
SHOT_SIZE = 12.0                    # objects smaller than this are projectiles
THREAT_PX = 70.0                    # threat (LC4) ramps up inside this distance
LOST_FRAMES = 30
FIXED_TICKS = 60                    # an object that hasn't moved a pixel in 3 s is part of the screen
                                    # (SMB's coin icon in the status bar), not something to chase or be
NO_RESPONSE_SECONDS = 3.0          # pressing directions this long and nothing on screen follows -> menu / title
DEMO_SECONDS = 9.0                  # left/right-only games: both directions ignored this long -> attract demo
FOLLOW_WINDOW = 100                 # recent moves of the tracked sprite while a direction is held...
FOLLOW_DOUBT = 0.8                  # ...fewer of them going that way than this: doubt. Playing: 0.9-0.99;
                                    # an attract demo that fooled the tracker (Galaga after game over): 0.35-0.75
WIGGLE_TICKS = 8                    # doubt -> press one way this many ticks, then the other way...
WIGGLE_PX = 4.0                     # ...and what we track must go at least this far each way (px)
WIGGLE_EVERY = 10.0                 # at most one check this often (s)
UNSTICK_HOLD = 2                    # ticks a menu key is held, alone, before release
MENU_QUIET = 0.5                    # seconds of nothing pressed before a menu key (Excitebike ignores START
                                    # right after a d-pad press)
MENU_WAITS = 3                      # times a menu key is held back because the probes seem to move the player
PROBE_TICKS = 10                    # while seeking, alternate left / right this often
CONTROL_MOVES = 3                   # moves of the tracked sprite per direction, in 2+ directions, to judge control...
CONTROL_AGREE = 0.6                 # ...and the share that must follow the d-pad (a demo that drives itself: ~0.5)
INERTIA_TICKS = 5                   # probing: after reversing, ticks not judged (Mario skids the old way)
SCROLL_LEVEL = 2.0                  # mean brightness change per tick (0-255) that counts as the screen moving
SCROLL_SHARE = 0.8                  # racing games: in control once the screen moves this often while on the gas
SCENE_CUT = 30.0                    # mean brightness change in one tick that means a different screen
STILL_LEVEL = 0.5                   # below this mean change the screen counts as standing still
STILL_SECONDS = 6.0                 # a still screen this long while "in control": results / map / pause screen
BORED_SECONDS = 20.0                # in control, but no new screen and no scrolling this long: an overworld
                                    # map (SMB3), or stuck against a wall (then "right, A" is a jump)
EXPLORE_STEPS = [("right", 4), (None, 6), ("a", 2), (None, 16), ("down", 4), (None, 6), ("a", 2), (None, 16),
                 ("up", 4), (None, 6), ("a", 2), (None, 16), ("left", 4), (None, 6), ("a", 2), (None, 16)]
#                   step on the map, then A to enter the level there (SMB3); (button, ticks), None = hands off
VAULT_SECONDS = 1.0                 # platformers: pushing forward this long without getting anywhere
VAULT_TICKS = (14, 3)               # ...then jump over it: forward + jump held (a high jump), then let go
VAULT_PROGRESS = 8.0                # px of forward movement that count as getting somewhere
WANDER_SECONDS = 3.0                # with nothing on screen, a faint target appears in a new direction this often
WANDER_GAIN = 1.0                   # chase drive of that target (1 = like a real object)
WANDER_FORWARD = 0.75               # share of wander targets placed in the game's forward direction
PAD_KEYS = ("a", "b", "select", "start", "up", "down", "left", "right")
OBJECT_FEATURES = 23                # learner inputs from object_features()
DECOR_MIN = 6                       # a tile drawn this many times as lone sprites is decoration (stars, snow)
# Eyes: real pixels -> the fly's elementary motion detectors (T4 = ON, T5 = OFF), looming only.
# The photoreceptor -> lamina -> medulla relays are skipped: they signal with graded voltages
# that a spiking model can't carry. From T4/T5 on, the connectome does the rest (-> LPLC2 -> DNp01...).
RETINA = (120, 128)                 # rows, cols: the screen downsampled 2x (sent by the page each tick)
EYE_GAIN = 30.0                     # growth of angular size per tick -> voltage per step
EYE_CAP = 0.6
EYE_SELF_PX = 14                    # pixels around the player's own sprite are not looked at (self-motion)
EYE_CUT = 0.08                      # mean brightness change above this is a scene cut, not motion
EYE_EDGE = 0.12                     # a pixel this far from the background brightness belongs to a thing
EYE_MATCH_PX = 10                   # a thing moves at most this far (retina pixels) between ticks
# Columnar motion (--motion columns): every T4/T5 cell at its own place in the eye, driven by
# motion in its own preferred direction (both read off the wiring, flybrain.columns).
COL_SPACING = (1, 2, 3)             # retina px between the two inputs of a motion detector (speeds)
COL_BLUR = 1.5                      # px: a column sees a small patch, not one pixel
COL_GAIN = 18.0                     # detector output -> voltage per step (at 6 looming barely reaches the DNs)
COL_CAP = 0.6
COL_EYE_SPAN = 0.6                  # each eye covers this share of the screen width (they overlap in the middle)


# ---------------------------------------------------------------- sprites -> objects

def cluster_sprites(sprites: list[dict], tall: bool = False, gap: float = CLUSTER_PX) -> list[dict]:
    """Group sprite tiles (8x8, or 8x16 when `tall`) into objects (connected within `gap` px).
    Decoration drawn with sprites, like Galaga's starfield, is left out."""
    sh = 16 if tall else 8
    seen = {(s["x"], s["y"]): s["tile"] for s in sprites
            if PLAY_TOP <= s["y"] <= PLAY_BOTTOM and 0 <= s["x"] <= SCREEN_W - 4}

    def lone(x, y):      # no tile beside it: ships are 2+ tiles wide, stars and snowflakes one
        return not any((x2, y2) != (x, y) and abs(x2 - x) <= gap and abs(y2 - y) <= 1 for x2, y2 in seen)

    by_tile: dict[int, list] = {}
    for xy, tile in seen.items():
        by_tile.setdefault(tile, []).append(xy)
    decor = {t for t, xys in by_tile.items()
             if len(xys) >= DECOR_MIN and sum(lone(*xy) for xy in xys) >= 0.8 * len(xys)}
    pts = sorted(xy for xy, tile in seen.items() if tile not in decor)
    label = list(range(len(pts)))

    def root(i):
        while label[i] != i:
            label[i] = label[label[i]]
            i = label[i]
        return i

    for i, (xi, yi) in enumerate(pts):
        for j in range(i + 1, len(pts)):
            xj, yj = pts[j]
            if xj - xi > gap:
                break
            if abs(yj - yi) <= max(gap, sh + 1):
                label[root(j)] = root(i)
    groups: dict[int, list] = {}
    for i, p in enumerate(pts):
        groups.setdefault(root(i), []).append(p)
    out = []
    for members in groups.values():
        xs = np.array([m[0] for m in members], np.float32)
        ys = np.array([m[1] for m in members], np.float32)
        w, h = float(xs.max() - xs.min() + 8), float(ys.max() - ys.min() + sh)
        if len(members) > 16 or max(w, h) > 48:      # HUD, text boxes, scenery built from sprites
            continue
        out.append({"x": float(xs.mean()) + 4, "y": float(ys.mean()) + sh / 2, "w": w, "h": h, "n": len(members)})
    return out


def merge_human(pad: dict[str, bool], held: dict, drive: bool) -> dict[str, bool]:
    """A person's buttons on top of the fly's. Each held button is pressed; a held direction
    wins its axis (left beats the fly's right). drive: the person alone plays."""
    out = {k: False for k in pad} if drive else dict(pad)
    held = {k for k, v in held.items() if v and k in PAD_KEYS}
    for axis in (("left", "right"), ("up", "down")):
        if held & set(axis):
            for k in axis:
                out[k] = False
    for k in held:
        out[k] = True
    return out


class Tracker:
    """Follow objects across ticks and find which one is the player: the object whose
    motion agrees with the directions we have been pressing."""

    def __init__(self):
        self.tracks: list[dict] = []
        self.next_id = 0
        self.player_id: int | None = None
        self.missed = 0
        self.responded = False
        self.player_agree = None             # did the player sprite follow the d-pad this tick (+1 / -1 / None)

    def update(self, objs: list[dict], pressed: tuple[int, int]) -> tuple[dict | None, list[dict]]:
        self.responded = False
        self.player_agree = None
        if not objs:
            self.missed += 1
            if self.missed > LOST_FRAMES:
                self.tracks, self.player_id = [], None
            player = next((t for t in self.tracks if t["id"] == self.player_id), None)
            return player, []
        self.missed = 0
        free = list(self.tracks)
        tracks = []
        for o in objs:
            best = min(free, key=lambda t: (t["x"] - o["x"]) ** 2 + (t["y"] - o["y"]) ** 2, default=None)
            if best is not None and np.hypot(best["x"] - o["x"], best["y"] - o["y"]) <= MATCH_PX:
                free.remove(best)
                dx, dy = o["x"] - best["x"], o["y"] - best["y"]
                agree = np.sign(dx) * pressed[0] + np.sign(dy) * pressed[1]
                score = 0.9 * best["score"] + (agree if any(pressed) else 0.0)
                if any(pressed) and agree > 0:
                    self.responded = True
                moved = any(pressed) and (dx or dy)
                tracks.append({**o, "id": best["id"], "score": score, "vx": dx, "vy": dy, "age": best["age"] + 1,
                               "agree": float(agree) if moved else None,
                               "still": 0 if (dx or dy) else best.get("still", 0) + 1})
            else:
                tracks.append({**o, "id": self.next_id, "score": 0.0, "vx": 0.0, "vy": 0.0, "age": 0, "still": 0})
                self.next_id += 1
        for t in free:                                       # not seen this tick (sprite flicker): keep briefly
            if t.get("miss", 0) < 5:
                tracks.append({**t, "miss": t.get("miss", 0) + 1})
        self.tracks = tracks
        best = max(tracks, key=lambda t: t["score"])
        current = next((t for t in tracks if t["id"] == self.player_id), None)
        if best["score"] > 1.5 and (current is None or best["score"] > current["score"] + 1.0):
            current = best                                   # clearly steerable: that's us
        if current is None:                                  # first guess: something alive, nearest the centre
            current = min(tracks, key=lambda t: (t["still"] >= FIXED_TICKS, (t["x"] - 128) ** 2 + (t["y"] - 120) ** 2))
        self.player_id = current["id"]
        self.player_agree = current.get("agree")
        others = [t for t in tracks if t["id"] != current["id"] and t["still"] < FIXED_TICKS]
        others.sort(key=lambda t: (t["x"] - current["x"]) ** 2 + (t["y"] - current["y"]) ** 2)
        return current, others[:6]


# ---------------------------------------------------------------- objects -> neurons

class TwoAxisEyes:
    """Drive LPLC2 / LC4 / LPLC1 / LC10a per fly, as seen along that fly's axis.

    For an object at (dx, dy) from the player, fly H puts it on its left if dx < 0,
    fly V on its left if dy < 0 (above). The amount is weighted by how much of the
    offset lies along the fly's axis, so a guard straight above barely registers
    for fly H and fully for fly V. Encoder parameters are flybrain.eyes.ENCODER."""

    def __init__(self, brain, batch: int, **encoder):
        unknown = set(encoder) - set(ENCODER)
        if unknown:
            raise ValueError(f"unknown encoder parameters: {sorted(unknown)}")
        self.p = {k: float(encoder.get(k, v)) for k, v in ENCODER.items()}
        self.cells = {ch: {s: brain.cells(types, s) for s in "LR"} for ch, types in CHANNELS.items()}
        self.axis_of = np.array([b % 2 for b in range(batch)])
        self.prev_angle: dict[int, float] = {}
        self.last = {f"{ch}{a}{s}": 0.0 for ch in CHANNELS for a in AXES for s in "LR"}

    def calibration(self, channels, side: str, amount: float) -> list:
        return [(self.cells[ch][side], np.float32(amount)) for ch in channels]

    def inject(self, player: dict | None, others: list[dict], wander: tuple[float, float] | None):
        p = self.p
        drive = {k: 0.0 for k in self.last}
        seen = {}

        def add(key, value):
            drive[key] = max(drive[key], float(np.clip(value, 0, p["cap"])))

        if player is not None:
            targets = [(o, False) for o in others]
            if not others and wander is not None:            # nothing to look at: a faint target to walk to
                wx, wy = wander
                targets = [({"id": -1, "x": player["x"] + wx, "y": player["y"] + wy, "w": 8.0, "h": 8.0}, True)]
            for rank, (o, faint) in enumerate(targets):
                dx, dy = o["x"] - player["x"], o["y"] - player["y"]
                dist = max(float(np.hypot(dx, dy)), 1.0)
                size = max(o["w"], o["h"])
                angle = size / max(dist, 8.0)
                growth = max(0.0, angle - self.prev_angle.get(o["id"], angle))
                seen[o["id"]] = angle
                shot = size < SHOT_SIZE and not faint
                for axis, comp in (("H", dx), ("V", dy)):
                    w = abs(comp) / dist                     # share of the offset along this axis
                    if w < 0.2:
                        continue
                    s = "L" if comp < 0 else "R"
                    if shot:
                        add(f"shot{axis}{s}", w * growth * p["shot_gain"])
                        continue
                    if rank == 0:                            # only the nearest body is chased
                        gain = WANDER_GAIN if faint else 1.0
                        add(f"chase{axis}{s}", w * gain * (p["chase_base"] + p["chase_gain"] * angle))
                    if not faint:                            # every body can loom and threaten
                        add(f"loom{axis}{s}", w * (growth * p["loom_gain"] + angle * p["loom_size"]))
                        add(f"threat{axis}{s}", w * p["threat_max"] * max(0.0, 1 - dist / THREAT_PX))
        self.prev_angle = seen
        self.last = drive
        out = []
        for ch in CHANNELS:
            for s in "LR":
                amount = np.array([drive[f"{ch}{AXES[a]}{s}"] for a in self.axis_of], np.float32)
                if amount.any():
                    out.append((self.cells[ch][s], amount))
        return out

    def display(self) -> dict[str, float]:
        """Dashboard bars: loomL/loomR... for fly H and loomU/loomD... for fly V."""
        d = {}
        for ch in CHANNELS:
            d[f"{ch}L"], d[f"{ch}R"] = self.last[f"{ch}HL"], self.last[f"{ch}HR"]
            d[f"{ch}U"], d[f"{ch}D"] = self.last[f"{ch}VL"], self.last[f"{ch}VR"]
        return d


class MotionEyes:
    """Pixels -> T4/T5 looming input, per fly axis and per eye.

    The picture is split into things that stand out from the background (connected
    patches of pixels). Each is followed from frame to frame; when its angular size
    (size / distance from the player) grows, it is expanding on the fly's retina, which
    is what LPLC2 detects. That growth drives the outward-motion cells on the side it is
    on (T4a if it is brighter than the background, T5a if darker), weighted by how much
    of its offset lies along the fly's axis. Things that only slide past or stand still
    are not passed on: in this model any T4/T5 input excites LPLC2, so they would read
    as looming. No sprite data is used here, only pixels."""

    def __init__(self, brain, batch: int):
        ct = brain.cell_type.astype(str)
        self.cells = {f"{kind}a{s}": np.flatnonzero((ct == f"{kind}a") & (brain.side == s))
                      for kind in ("T4", "T5") for s in "LR"}
        self.axis_of = np.array([b % 2 for b in range(batch)])
        self.blobs: list[tuple[float, float, float]] = []   # (x, y, angular size) last frame
        self.prev: np.ndarray | None = None
        self.last: dict[str, float] = {}

    def reset(self) -> None:
        self.prev, self.blobs = None, []

    def calibration(self, channels, side: str, amount: float) -> list:
        """Expansion on one side: the outward-motion cells, ON and OFF."""
        return [(self.cells[f"{kind}a{side}"], np.float32(amount)) for kind in ("T4", "T5")]

    def inject(self, pixels: np.ndarray | None, player: dict | None) -> list:
        from scipy import ndimage
        self.last = {}
        if pixels is None:
            self.prev, self.blobs = None, []
            return []
        img = pixels.astype(np.float32) / 255.0
        prev, self.prev = self.prev, img
        if prev is None or prev.shape != img.shape or np.abs(img - prev).mean() > EYE_CUT:
            self.blobs = []
            return []                                        # first frame, or a scene cut (new room, flash)
        rows, cols = img.shape
        sy, sx = rows / SCREEN_H, cols / SCREEN_W
        px = player["x"] * sx if player else cols / 2
        py = player["y"] * sy if player else rows / 2
        bg = float(np.median(img))
        fg = np.abs(img - bg) > EYE_EDGE
        labels, n = ndimage.label(fg)
        found = ndimage.find_objects(labels)
        blobs, drive = [], {f"{k}{a}{s}": 0.0 for k in ("T4", "T5") for a in AXES for s in "LR"}
        for i, sl in enumerate(found, start=1):
            m = labels[sl] == i
            area = int(m.sum())
            if area < 4 or area > 0.25 * rows * cols:        # specks, or the scenery itself
                continue
            ys, xs = np.nonzero(m)
            cx, cy = xs.mean() + sl[1].start, ys.mean() + sl[0].start
            dx, dy = cx - px, cy - py
            if abs(dx) < EYE_SELF_PX * sx and abs(dy) < EYE_SELF_PX * sy:
                continue                                     # the player itself
            dist = max(float(np.hypot(dx, dy)), 2.0)
            angle = np.sqrt(area) / dist
            blobs.append((cx, cy, angle))
            near = [b for b in self.blobs if np.hypot(b[0] - cx, b[1] - cy) <= EYE_MATCH_PX]
            if not near:
                continue
            before = min(near, key=lambda b: np.hypot(b[0] - cx, b[1] - cy))[2]
            growth = angle - before
            if growth <= 0:
                continue                                     # not getting bigger on the retina
            kind = "T4" if img[sl][m].mean() > bg else "T5"
            for axis, comp in (("H", dx), ("V", dy)):
                w = abs(comp) / dist
                if w >= 0.2:
                    drive[f"{kind}{axis}{'L' if comp < 0 else 'R'}"] += w * growth
        self.blobs = blobs
        out = []
        for kind in ("T4", "T5"):
            for s in "LR":
                v = [drive[f"{kind}{AXES[a]}{s}"] for a in self.axis_of]
                amount = np.clip(EYE_GAIN * np.array(v, np.float32), 0, EYE_CAP)
                self.last[f"{kind}a{s}"] = float(amount.mean())
                if amount.any():
                    out.append((self.cells[f"{kind}a{s}"], amount))
        return out


def eye_layout() -> dict:
    """Every T4/T5 cell placed on the screen: the left eye sees the left 60%, the right eye the
    right 60%, fronts toward the middle, tops up (an assumed viewing layout). x, y in 0..1 of the
    screen; dir = the screen direction (x right, y down) the cell prefers."""
    from flybrain.columns import eye_map
    m = eye_map()
    left = m["eye"] == "L"
    front, up = m["front"], m["up"]
    side = np.where(left, 1.0, -1.0)                         # a step to the front is rightward in the left eye
    return {"cells": m["cells"], "left": left, "types": m["types"].astype(str),
            "x": np.where(left, COL_EYE_SPAN * front, 1 - COL_EYE_SPAN * front).astype(np.float32),
            "y": (1 - up).astype(np.float32),
            "dir": np.column_stack([m["pref"][:, 0] * side, -m["pref"][:, 1]]).astype(np.float32)}


class ColumnEyes:
    """Pixels -> every T4 (ON) and T5 (OFF) cell, by its own column and preferred direction.

    The eye map (flybrain.columns) places each cell in its eye and gives the direction it
    prefers, both from the wiring alone. The left eye sees the left 60% of the screen, the
    right eye the right 60%, fronts toward the middle, tops up (an assumed viewing layout,
    not a measured one). Per tick the picture's brightening (ON) and darkening (OFF) edges
    are correlated across neighbouring points one tick apart (Hassenstein-Reichardt: the
    delay-and-multiply that T4/T5 compute from their Mi/Tm inputs), for rightward, leftward,
    downward and upward motion; each cell gets the motion along its own preferred direction
    at its own place. Fly V sees the picture turned (above = its left), as elsewhere.
    What happens next - looming in LPLC2, wide-field flow in HS/VS - is the connectome's."""

    def __init__(self, brain, batch: int):
        from flybrain.columns import eye_map
        lay = eye_layout()
        self.cells, self.x, self.y, self.dir = lay["cells"], lay["x"], lay["y"], lay["dir"]
        self.types, self.left = lay["types"], lay["left"]
        self.on = np.char.startswith(self.types, "T4")
        self.axis_of = np.array([b % 2 for b in range(batch)])
        self.prev: dict[int, np.ndarray] = {}
        self.prev_edges: dict[int, tuple] = {}
        self.last: dict[str, float] = {}

    def reset(self) -> None:
        self.prev, self.prev_edges = {}, {}

    @staticmethod
    def _correlate(a0: np.ndarray, a1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(horizontal, vertical) motion of one polarity between two ticks: + = right / down."""
        h = np.zeros_like(a1)
        v = np.zeros_like(a1)
        for d in COL_SPACING:
            r = a0[:, :-d] * a1[:, d:] - a1[:, :-d] * a0[:, d:]   # here before, there now: moved right
            h[:, :-d] += r
            h[:, d:] += r
            c = a0[:-d] * a1[d:] - a1[:-d] * a0[d:]
            v[:-d] += c
            v[d:] += c
        return h, v

    def _drive(self, img: np.ndarray, view: int) -> np.ndarray | None:
        from scipy import ndimage
        prev = self.prev.get(view)
        self.prev[view] = img
        if prev is None or prev.shape != img.shape:
            self.prev_edges.pop(view, None)
            return None
        d = img - prev
        on, off = np.maximum(d, 0), np.maximum(-d, 0)
        before = self.prev_edges.get(view)
        self.prev_edges[view] = (on, off)
        if before is None:
            return None
        rows, cols = img.shape
        r = np.minimum((self.y * (rows - 1)).round().astype(int), rows - 1)
        c = np.minimum((self.x * (cols - 1)).round().astype(int), cols - 1)
        out = np.zeros(len(self.cells), np.float32)
        for pol, now, then in ((True, on, before[0]), (False, off, before[1])):
            h, v = (ndimage.gaussian_filter(m, COL_BLUR) for m in self._correlate(then, now))
            m = self.on == pol
            out[m] = h[r[m], c[m]] * self.dir[m, 0] + v[r[m], c[m]] * self.dir[m, 1]
        return np.maximum(out, 0)

    def _views(self, pixels: np.ndarray, player: dict | None) -> dict[int, np.ndarray]:
        img = pixels.astype(np.float32) / 255.0
        if player is not None:                               # its own sprite: self-motion, not looked at
            rows, cols = img.shape
            x, y = player["x"] * cols / SCREEN_W, player["y"] * rows / SCREEN_H
            dx, dy = EYE_SELF_PX * cols / SCREEN_W, EYE_SELF_PX * rows / SCREEN_H
            img = img.copy()
            img[max(int(y - dy), 0):int(y + dy), max(int(x - dx), 0):int(x + dx)] = np.median(img)
        return {0: img, 1: np.ascontiguousarray(img.T)}      # H flies: as it is; V flies: turned

    def calibration(self, channels, side: str, amount: float) -> list:
        """Through the same detectors and gain as in play. motion: looming on one side (a dark
        disc growing on a grey screen); flow: a textured screen sliding toward that side."""
        from scipy import ndimage
        saved = self.prev, self.prev_edges
        self.prev, self.prev_edges = {}, {}
        rows, cols = RETINA
        yy, xx = np.mgrid[:rows, :cols]
        cx = cols * (0.25 if side == "L" else 0.75)
        texture = ndimage.gaussian_filter(np.random.default_rng(7).random((rows, 2 * cols)), 2.0)
        texture = (0.5 + 0.15 * (texture - texture.mean()) / texture.std()).clip(0, 1).astype(np.float32)
        step = -2 if side == "L" else 2
        total, n = np.zeros(len(self.cells), np.float32), 0
        for k, radius in enumerate(np.geomspace(3, 24, 8)):
            if channels == ("flow",):
                img = texture[:, (np.arange(cols) - step * k) % texture.shape[1]]
            else:
                img = np.where(np.hypot(xx - cx, yy - rows / 2) < radius, 0.1, 0.6).astype(np.float32)
            d = self._drive(img, 0)
            if d is not None:
                total += d
                n += 1
        self.prev, self.prev_edges = saved
        drive = np.clip(COL_GAIN * total / max(n, 1) * amount / CAL_AMOUNT, 0, COL_CAP)
        return [(self.cells, np.repeat(drive[:, None], len(self.axis_of), axis=1))]   # same for every fly

    def inject(self, pixels: np.ndarray | None, player: dict | None) -> list:
        self.last = {}
        if pixels is None:
            self.prev, self.prev_edges = {}, {}
            return []
        prev = self.prev.get(0)
        img = pixels.astype(np.float32) / 255.0
        if prev is not None and prev.shape == img.shape and np.abs(img - prev).mean() > SCENE_CUT / 255:
            self.prev, self.prev_edges = {}, {}                  # a scene cut, not motion (scrolling is motion)
        drives = {view: self._drive(v, view) for view, v in self._views(pixels, player).items()}
        if any(d is None for d in drives.values()):
            return []
        amount = np.clip(COL_GAIN * np.stack([drives[a] for a in self.axis_of], axis=1), 0, COL_CAP)
        for t in ("T4a", "T4b", "T5a", "T5b"):
            for s, m in (("L", self.left), ("R", ~self.left)):
                self.last[f"{t}{s}"] = float(amount[(self.types == t) & m, 0].mean())
        return [(self.cells, amount)] if amount.any() else []


# ---------------------------------------------------------------- neurons -> pad

class Decoder:
    """All descending neurons -> NES pad, one fly (or voting group) per axis.

    Each readout is a weighted sum over the descending neurons that, in this brain,
    fire differently for a stimulus on the left than on the right. Weights come from
    FlyNes.calibrate(): positive score = the brain reacts as to something on its left."""

    def __init__(self, brain, batch: int, readouts: dict | None = None):
        self.readouts = readouts or READOUTS
        self.dn = brain.cells(["descending_neuron"])
        self.names = [str(brain.cell_type[i]) for i in self.dn]
        D = len(self.dn)
        self.lut = np.full(brain.n + 1, D, np.int64)               # neuron -> column (D = not a DN)
        self.lut[self.dn] = np.arange(D)
        self.axis_of = np.array([b % 2 for b in range(batch)])
        self.history: deque[np.ndarray] = deque(maxlen=max(w for _, w, _ in self.readouts.values()))
        self.rest = np.zeros((batch, D), np.float32)
        self.alpha = brain.dt / BASELINE_TAU
        self.w = {r: np.zeros(D, np.float32) for r in self.readouts}
        self.bias = dict.fromkeys(self.readouts, 0.0)
        self.threshold = dict.fromkeys(self.readouts, 1.0)
        self.top: dict[str, list[str]] = {}
        self.last = {"attack": -10.0}
        self.hold = {"attack": 0}
        self.state: dict = {}
        self.scores = {r: dict.fromkeys(AXES, 0.0) for r in self.readouts}

    def observe(self, flies: list[np.ndarray], learn_rest: bool = True) -> None:
        D = len(self.dn)
        counts = np.array([np.bincount(self.lut[f], minlength=D + 1)[:D] for f in flies], np.float32)
        self.history.append(counts)
        if learn_rest:
            self.rest += self.alpha * (counts - self.rest)

    def excess(self, window: int) -> np.ndarray:
        """(batch, DNs): spikes above resting rate over the last `window` steps."""
        h = list(self.history)[-window:]
        return np.sum(h, axis=0) - len(h) * self.rest

    def fit(self, name: str, X: np.ndarray, y: np.ndarray, noise: np.ndarray) -> None:
        """X: (samples, DNs) excess spikes, y: +1 stimulus left / -1 right; noise: no stimulus."""
        L, R = X[y > 0], X[y < 0]
        diff = L.mean(0) - R.mean(0)
        var = 0.5 * (L.var(0) + R.var(0)) + 0.25
        dprime = diff / np.sqrt(var)
        keep = np.argsort(-np.abs(dprime))[:TOP_NEURONS]
        w = np.zeros_like(diff)
        w[keep] = diff[keep] / var[keep]
        sL, sR = L @ w, R @ w
        scale = 2.0 / max(float(sL.mean() - sR.mean()), 1e-6)   # left stimulus -> +1, right -> -1
        self.w[name] = (w * scale).astype(np.float32)
        self.bias[name] = float(0.5 * (sL.mean() + sR.mean()) * scale)
        sd = float((noise @ self.w[name] - self.bias[name]).std())
        self.threshold[name] = max(THRESHOLD_SD * sd, MIN_THRESHOLD[name])
        self.top[name] = [self.names[i] + ("L" if diff[i] > 0 else "R") for i in keep[:5]]
        sep = float(sL.mean() - sR.mean()) / max(float(np.sqrt(0.5 * (sL.var() + sR.var()))), 1e-6)
        print(f"  {name:5s}: {int((np.abs(dprime) > 1).sum()):3d} side-selective DNs, top "
              f"{', '.join(self.top[name])}; L/R separation d'={sep:.1f}, threshold {self.threshold[name]:.2f}",
              flush=True)

    def per_axis(self, values: np.ndarray) -> dict[str, float]:
        return {AXES[a]: float(values[self.axis_of == a].mean()) for a in (0, 1)}

    def command(self, now: float, target: dict | None) -> dict[str, bool]:
        pad = dict.fromkeys(PAD_KEYS, False)
        cache: dict[int, np.ndarray] = {}
        for name, (_, window, _) in self.readouts.items():
            if window not in cache:
                cache[window] = self.excess(window)
            self.scores[name] = self.per_axis(cache[window] @ self.w[name] - self.bias[name])
        acting = dict.fromkeys(AXES, "")
        for axis in AXES:
            lo, hi = SIDES[axis]
            for name in PRIORITY:
                v = self.scores[name][axis]
                if abs(v) < self.threshold[name]:
                    continue
                left = v > 0                                       # the brain reacts to its left side
                toward = self.readouts[name][2] == "toward"
                pad[lo if left == toward else hi] = True
                acting[axis] = name
                break
        fleeing = any(a in ("flee", "looming", "dodge") for a in acting.values())
        # Strike when a target is within reach and either the pursuit readout points at it,
        # or the escape readout bursts (startle strike: in sshfighter the escape neurons were
        # the best predictor of a punch landing).
        facing = False
        if target is not None:
            for axis, comp in (("H", target["dx"]), ("V", target["dy"])):
                v = self.scores["chase"][axis]
                if abs(comp) >= 4 and abs(v) >= self.threshold["chase"] and (v > 0) == (comp < 0):
                    facing = True
        striking = target is not None and target["dist"] <= ATTACK_RANGE and (facing or bool({"flee", "looming"} & set(acting.values())))
        if striking and now - self.last["attack"] >= COOLDOWN["attack"]:
            self.hold["attack"], self.last["attack"] = HOLD_TICKS["attack"], now
        if self.hold["attack"] > 0:
            pad["a"] = True
            self.hold["attack"] -= 1
        self.state = {"acting": acting, "facing": facing, "fleeing": fleeing}
        return pad

    def snapshot(self) -> dict[str, float]:
        """Dashboard bars: each readout's lean per axis, 1.0 = its action threshold."""
        out = {}
        for name in self.readouts:
            for axis in AXES:
                v = self.scores[name][axis] / self.threshold[name]
                out[f"{name}{axis}_L"], out[f"{name}{axis}_R"] = max(v, 0.0), max(-v, 0.0)
        return out


# ---------------------------------------------------------------- the loop

def checkpoint_name(rom_name: str, motion: str, learner: str = "mc") -> str:
    """Readout calibration and learning depend on how the eyes feed the brain and on the learner:
    one file each, so trying an option never touches the default's checkpoint."""
    tags = ([] if motion == "blobs" else [motion]) + ([] if learner == "mc" else [learner])
    return ".".join([rom_name, *tags, "npz"])


class FlyNes:
    def __init__(self, device: str | None, voters: int, seed: int, encoder: dict | None, vision: str = "both",
                 checkpoint: Path | None = None, rom_name: str = "", motion: str = "blobs", learner: str = "mc"):
        print("loading connectome...", flush=True)
        self.batch = 2 * voters
        self.brain = FlyBrain(device=device, batch=self.batch, seed=seed)
        self.eyes = TwoAxisEyes(self.brain, self.batch, **(encoder or {}))
        # pixels -> T4/T5: "blobs" (things growing on the retina -> T4a/T5a) or "columns" (every
        # T4/T5 cell by its own column and preferred direction; see ColumnEyes)
        self.motion = ColumnEyes(self.brain, self.batch) if motion == "columns" else MotionEyes(self.brain, self.batch)
        self.motion_kind = motion
        # "mc": 1 s Monte Carlo on the brain's readouts (default). "td": n-step TD + forgetting + object
        # positions - tested against mc (Galaga, Mario, 3 seeds x 15 min) and not better, so opt-in
        self.learner_kind = learner
        self.readouts = {**READOUTS, **FLOW_READOUT} if motion == "columns" else READOUTS
        self.vision = vision            # "sprites": feature detectors only, "eyes": pixels -> T4/T5 only, "both"
        self.monitor = None              # BrainMonitor, set by the dashboard
        self.stim_gain: dict[str, float] = {}            # per sense, x calibrated level (reward_feed.STIMULUS)
        self.decoder = Decoder(self.brain, self.batch, self.readouts)
        self.tracker = Tracker()
        self.rng = np.random.default_rng(seed)
        self.game_time = 0.0
        self.step_ms = 0.0
        self.last_sprites: list[dict] = []
        self.last_player: dict | None = None
        self.last_others: list[dict] = []
        self.last_frame = 0
        self.frame_spikes = np.empty(0, np.int64)
        self.in_control = False          # has anything on screen followed our d-pad yet?
        self.unresponsive = 0.0          # seconds of pressing directions with nothing following
        self.tried: set[str] = set()     # directions pressed in that time (a wall blocks only some)
        self.agreement: dict[str, deque] = {}       # per pressed direction: did the player sprite follow it?
        self.unstick_left = 0
        self.probe_tick = 0
        self.prev_pixels: np.ndarray | None = None
        self.scrolling: deque = deque(maxlen=40)   # 2 s: was the screen moving, tick by tick
        self.last_change = 0.0
        self.still_for = 0.0
        self.since_new = 0.0                 # seconds since a new screen or scrolling
        self.explore: list | None = None
        self.push = None                     # (start position, seconds) of pushing forward with no progress
        self.press_key, self.press_run = None, 0     # the one direction held, and for how many ticks
        self.reversed = False                        # ...reached by reversing the previous one
        self.recent_follow: deque = deque(maxlen=FOLLOW_WINDOW)
        self.wiggle: dict | None = None                  # pressing one way, then the other: is it ours?
        self.next_wiggle = 0.0
        self.lost_control = False
        self.menu_waits = 0
        self.vault_left = 0
        self.race_check: dict | None = None        # checking whether a scrolling screen is a race or a demo
        self.race_block_until = 0.0
        self.race_ok = False
        self.next_unstick = 0.0
        self.unstick_i = 0
        self.last_pad = dict.fromkeys(PAD_KEYS, False)
        self.wander: tuple[float, float] | None = None
        self.wander_until = 0.0
        self.seeking = True
        print(f"brain ready: {self.brain.n:,} neurons, {len(self.brain.indices):,} connections on "
              f"{self.brain.device}, {self.batch} flies ({voters} per axis)", flush=True)
        for _ in range(int(BASELINE_TAU / self.brain.dt)):   # settle, and learn each neuron's resting rate
            self.decoder.alpha = 5 * self.brain.dt / BASELINE_TAU
            self.decoder.observe(self._step([]))
        self.decoder.alpha = self.brain.dt / BASELINE_TAU
        saved = self._saved_calibration(checkpoint)
        if saved:
            self._use_calibration(saved)
            print(f"readout calibration loaded from {checkpoint.name}", flush=True)
        else:
            self.calibrate()
        self.learner = None
        self.checkpoint_dir = checkpoint.parent if checkpoint is not None else None
        self._setup_game(rom_name, checkpoint)
        self.game_time = self.brain.steps * self.brain.dt

    def _setup_game(self, rom_name: str, checkpoint: Path | None) -> None:
        """Hold the controller the way this game wants, and learn in this game's checkpoint."""
        self.controls = controls_for(rom_name)
        c = self.controls
        self.stim_gain = load_reward_config(rom_name, c.get("genre", ""))["stimulus"]
        print(f"game: {c['game']} ({c['genre']}) - steer {c['axes'] or 'none'}, fire {c['fire']}, "
              f"jump {c['jump']}, hold {list(c['hold'])}" + ("" if c["ok"] else "  [not suited to a fly: expect little]"),
              flush=True)
        self.learner = None
        self.reward = None
        self.finder = None
        self.checkpoint, self.rom_name = checkpoint, rom_name
        dec = self.decoder
        self.feat_idx = np.flatnonzero(np.any([w != 0 for w in dec.w.values()], axis=0))
        if checkpoint is not None:
            profile = profile_for(rom_name) or Learner.saved_profile(checkpoint)   # known RAM map first
            if profile:
                self._start_learning(profile)
            else:
                self.finder = RewardFinder()
                print("learning: looking for score and lives in RAM (starts after ~2 min of play)", flush=True)

    def switch_game(self, rom_name: str) -> dict:
        """The player loaded another game: save what was learned, then adapt to the new one -
        its controls, its checkpoint (resumed) or a fresh score/lives search - and forget
        everything about the old screen (which sprite is the player, menus, eyes)."""
        if rom_name == self.rom_name:
            return self.controls
        self.close()
        checkpoint = (self.checkpoint_dir / checkpoint_name(rom_name, self.motion_kind, self.learner_kind)
                      if self.checkpoint_dir else None)
        saved = self._saved_calibration(checkpoint)
        if saved:                                            # readouts that game's learning was built on
            self._use_calibration(saved)
        self._setup_game(rom_name, checkpoint)
        self.tracker = Tracker()
        self.in_control, self.unresponsive, self.seeking = False, 0.0, True
        self.tried.clear()
        self.agreement.clear()
        self.wander = None
        self.last_frame = 0
        self.last_pad = dict.fromkeys(PAD_KEYS, False)
        self.eyes.prev_angle = {}
        self.motion.reset()
        self.scrolling.clear()
        self.prev_pixels, self.race_check, self.race_block_until, self.still_for = None, None, 0.0, 0.0
        self.since_new, self.explore = 0.0, None
        self.push, self.vault_left = None, 0
        self.press_key, self.press_run, self.reversed, self.menu_waits = None, 0, False, 0
        self.recent_follow.clear()
        self.wiggle, self.next_wiggle, self.lost_control = None, 0.0, False
        return self.controls

    def calibrate(self) -> None:
        """Show each visual channel on the left and on the right, and learn which
        descending neurons report the side (the brain's own lateralized output)."""
        t0 = time.perf_counter()
        print("calibrating readouts on the descending neurons...", flush=True)
        dec = self.decoder
        for name, (channels, window, _) in self.readouts.items():
            X, y, noise = [], [], []
            for _ in range(CAL_REPS):
                for side, label in (("L", 1), ("R", -1), (None, 0)):
                    for _ in range(CAL_SETTLE):
                        dec.observe(self._step([]), learn_rest=False)
                    source = self.motion if channels in (("motion",), ("flow",)) else self.eyes
                    inject = [] if side is None else source.calibration(channels, side, CAL_AMOUNT)
                    for _ in range(window):
                        dec.observe(self._step(inject), learn_rest=False)
                    ex = dec.excess(window)
                    if side is None:
                        noise.extend(ex)
                    else:
                        X.extend(ex)
                        y.extend([label] * len(ex))
            dec.fit(name, np.array(X), np.array(y), np.array(noise))
        print(f"calibrated in {time.perf_counter() - t0:.1f} s", flush=True)

    def _start_learning(self, profile: dict) -> None:
        td = self.learner_kind == "td"
        self.learner = Learner(len(self.feat_idx) + len(self.readouts) + (OBJECT_FEATURES if td else 0) + 1,
                               self.checkpoint, self.rom_name, heads_for(self.controls), td=td)
        self.learner.extra = self._calibration_arrays()
        self.learner.profile = profile
        if self.learner.load():
            print(f"learning: resumed {self.checkpoint.name} ({self.learner.samples:,} samples, "
                  f"total reward {self.learner.total_reward:.0f})", flush=True)
        else:
            print(f"learning: new checkpoint {self.checkpoint}", flush=True)
        self.learner.save()
        self.reward = RewardFeed(profile, self.controls, self.rom_name, self.brain)   # events -> reward (reward_feed.py)
        self.finder = None

    def _calibration_arrays(self) -> dict:
        dec = self.decoder
        out = {}
        for name in self.readouts:
            out[f"w_{name}"] = dec.w[name]
            out[f"b_{name}"] = np.array([dec.bias[name], dec.threshold[name]])
        return out

    def _saved_calibration(self, checkpoint: Path | None) -> dict | None:
        if checkpoint is None or not checkpoint.is_file():
            return None
        z = np.load(checkpoint, allow_pickle=False)
        cal = {k[6:]: z[k] for k in z.files if k.startswith("extra_")}
        return cal if all(f"w_{r}" in cal for r in self.readouts) else None

    def _use_calibration(self, cal: dict) -> None:
        dec = self.decoder
        for name in self.readouts:
            dec.w[name] = cal[f"w_{name}"].astype(np.float32)
            dec.bias[name], dec.threshold[name] = (float(v) for v in cal[f"b_{name}"])
            top = np.argsort(-np.abs(dec.w[name]))[:5]
            dec.top[name] = [dec.names[i] + ("L" if dec.w[name][i] > 0 else "R") for i in top]

    def features_for_learning(self) -> dict[str, np.ndarray]:
        """Per learner head: the readouts' descending neurons (spikes over rest, last 0.1 s),
        the readout scores, where the nearest things are (object_features), and a constant."""
        dec = self.decoder
        ex = dec.excess(5)[:, self.feat_idx]
        objects = self.object_features() if self.learner_kind == "td" else np.zeros(0)
        out = {}
        for head, axes in (("H", (0,)), ("V", (1,)), ("fire", (0, 1))):
            m = np.isin(dec.axis_of, axes)
            scores = [np.mean([dec.scores[r][AXES[a]] for a in axes]) for r in self.readouts]
            out[head] = np.concatenate([ex[m].mean(axis=0), scores, objects, [1.0]])
        return out

    def object_features(self) -> np.ndarray:
        """The three nearest tracked objects relative to the player (offset, distance, size) and
        the player's place on screen: 23 numbers, the same for every game."""
        p, others = self.last_player, self.last_others[:3]
        v = []
        for k in range(3):
            if p is not None and k < len(others):
                o = others[k]
                dx, dy = (o["x"] - p["x"]) / 128, (o["y"] - p["y"]) / 120
                v += [dx, dy, abs(dx), abs(dy), 1 / (1 + 8 * np.hypot(dx, dy)), o["w"] * o["h"] / 256, 1.0]
            else:
                v += [0.0] * 7
        v += [0.0 if p is None else p["x"] / 256, 0.0 if p is None else p["y"] / 240]
        return np.array(v, np.float64)

    def close(self) -> None:
        if self.learner is not None:
            self.learner.profile = self.reward.p
            self.learner.save()
            print(f"\nlearning: saved {self.learner.path} ({self.learner.samples:,} samples)", flush=True)

    @property
    def features(self):          # the dashboard reads .cells / .last from here
        return self.eyes

    def _step(self, inject) -> list[np.ndarray]:
        fired = self.brain.step(None, inject)
        return fired if isinstance(fired, list) else [fired]

    def react(self, sprites: list[dict], frame: int = 0, tall: bool = False,
              pixels: np.ndarray | None = None, ram: np.ndarray | None = None,
              objects: list[dict] | None = None, human: dict | None = None) -> dict[str, bool]:
        """human: buttons a person holds on the dashboard pad ({"buttons": {...}, "drive": bool})."""
        gap = frame - self.last_frame
        if gap == 0 and frame:
            return self.last_pad                              # emulator hasn't advanced (paused / throttled tab)
        if gap < 0 or frame == 0:                            # new ROM or reload
            self.in_control, self.unresponsive = False, 0.0
        gap = gap if 0 < gap <= 30 else 1
        self.game_time += gap / 60
        self.last_frame = frame
        self.last_sprites = sprites
        pressed = (int(self.last_pad["right"]) - int(self.last_pad["left"]),
                   int(self.last_pad["down"]) - int(self.last_pad["up"]))
        objs = objects if objects is not None else cluster_sprites(sprites, tall)
        player, others = self.tracker.update(objs, pressed)
        self.last_player, self.last_others = player, others

        now = self.brain.steps * self.brain.dt
        if others or player is None:
            self.wander = None
        elif self.wander is None or now >= self.wander_until:
            self.wander = self._wander_target()
            self.wander_until = now + WANDER_SECONDS
        inject = []
        if self.vision in ("sprites", "both"):
            inject += self.eyes.inject(player, others, self.wander)
        if self.vision in ("eyes", "both"):
            inject += self.motion.inject(pixels, player)
        if any(g != 1.0 for g in self.stim_gain.values()):
            inject = self._scale_senses(inject)
        if self.reward is not None:
            inject += self.reward.taste_inject()          # rewards tasted: sweet + PAM, bitter + PPL1 (if on)

        t0 = time.perf_counter()
        spikes = []
        tick_flies = []
        steps = 0
        while self.brain.steps * self.brain.dt < self.game_time and steps < 4:
            flies = self._step(inject)
            self.decoder.observe(flies, learn_rest=not inject)   # resting rate: only while shown nothing
            spikes.append(flies[0])
            tick_flies.append(flies)
            steps += 1
        self.frame_spikes = np.concatenate(spikes) if spikes else np.empty(0, np.int64)
        if self.brain.steps * self.brain.dt < self.game_time - 0.1:
            self.game_time = self.brain.steps * self.brain.dt   # fell behind: skip ahead
        self.step_ms = 0.9 * self.step_ms + 0.1 * (time.perf_counter() - t0) * 1000

        target = None
        if player is not None and others:
            o = others[0]
            dx, dy = o["x"] - player["x"], o["y"] - player["y"]
            target = {"dx": dx, "dy": dy, "dist": float(np.hypot(dx, dy))}
        pad = self.decoder.command(now, target=target)
        pad = self._apply_controls(pad)
        if pixels is not None:
            if self.prev_pixels is not None and self.prev_pixels.shape == pixels.shape:
                change = float(np.abs(pixels.astype(np.int16) - self.prev_pixels).mean())
                self.scrolling.append(change > SCROLL_LEVEL)
                self.last_change = change
                self.still_for = self.still_for + gap / 60 if change < STILL_LEVEL else 0.0
                moving = len(self.scrolling) == self.scrolling.maxlen and np.mean(self.scrolling) >= 0.5
                self.since_new = 0.0 if (change > SCENE_CUT or moving) else self.since_new + gap / 60
            self.prev_pixels = pixels.astype(np.int16)
        if self.tracker.responded:           # the player follows the pad: playing, however little of the
            self.still_for = 0.0             # screen changes (small Mario walking on a still level)
        pad = self._menus(pad, now, gap / 60, any(pressed))
        if self.finder is not None:
            found = self.finder.observe(ram, player_lost=player is None, playing=not self.seeking)
            if found:
                print(f"\nlearning: found score {found['score']} and lives "
                      f"{hex(found['lives']) if found['lives'] is not None else 'not found'} in RAM", flush=True)
                self._start_learning(found)
        if self.learner is not None:
            r = self.reward(ram, self, gap / 60)
            if not self.seeking and self.wiggle is None:     # (a control check drives the pad itself)
                pad = self.learner.step(self.features_for_learning(), pad, r)
        if not self.seeking and self.wiggle is None:
            pad = self._vault(pad, player, gap / 60)
        if human:
            pad = merge_human(pad, human.get("buttons") or {}, bool(human.get("drive")))
            if self.learner is not None:
                self.learner.relabel(pad)            # credit the game's reward to what was really pressed
        self.last_pad = pad
        if self.monitor is not None and self.monitor.watched():
            self.monitor.tick(inject, tick_flies, pad, pixels)   # the brain view (brain_monitor.py)
        return pad

    def _scale_senses(self, inject: list) -> list:
        """The user's per-sense gains (rewards page): each injected set scaled by its sense's gain."""
        if not hasattr(self, "_sense_of"):
            self._sense_of = {id(a): ch for ch, d in self.eyes.cells.items() for a in d.values()}
            cells = self.motion.cells
            for a in (cells.values() if isinstance(cells, dict) else [cells]):
                self._sense_of[id(a)] = "eyes"
        out = []
        for idx, amt in inject:
            g = self.stim_gain.get(self._sense_of.get(id(idx), ""), 1.0)
            out.append((idx, amt * np.float32(g)) if g != 1.0 else (idx, amt))
        return out

    def _apply_controls(self, pad: dict[str, bool]) -> dict[str, bool]:
        """Hold the controller the way this game wants (games.py): only its d-pad axes,
        its fire button for strikes, its jump button when an escape readout fires."""
        c = self.controls
        strike = pad["a"]
        out = dict.fromkeys(PAD_KEYS, False)
        for axis in c["axes"]:
            for k in SIDES[axis]:
                out[k] = pad[k]
        if strike and c["fire"]:
            out[c["fire"]] = True
        acting = set(self.decoder.state.get("acting", {}).values())
        if c["jump"] and acting & {"flee", "looming", "dodge"}:
            out[c["jump"]] = True                           # the giant fibre's take-off, as a jump
        for k in c["hold"]:
            out[k] = True
        return out

    def _menus(self, pad: dict[str, bool], now: float, dt: float, was_pressing: bool) -> dict[str, bool]:
        """Title screens, pause menus, attract demos and radio calls ignore the d-pad. Detect that
        (the tracked sprite does not follow our presses) and tap START / SELECT / B / A, each alone,
        until a sprite follows the d-pad again."""
        a = self.tracker.player_agree
        pressed = [k for k in ("up", "down", "left", "right") if self.last_pad[k]]
        key = pressed[0] if len(pressed) == 1 else None
        if key != self.press_key:
            self.reversed = key is not None and self.press_key == OPPOSITE[key]
            self.press_run = 0
        self.press_run += key is not None
        self.press_key = key
        # Momentum: right after a probe reverses direction the player may still slide the old way
        # (Mario skids). Only while probing - in play the brain changes direction every few ticks.
        skid = self.seeking and self.reversed and self.press_run <= INERTIA_TICKS
        if a and key and not skid:
            self.agreement.setdefault(key, deque(maxlen=10)).append(a > 0)
            self.recent_follow.append(a > 0)
        # it must follow the d-pad in two or more directions: something drifting one way on its
        # own (title animations, a demo) can't
        good = [d for d, q in self.agreement.items() if len(q) >= CONTROL_MOVES and np.mean(q) >= CONTROL_AGREE]
        follows = len(good) >= 2
        # Racing games scroll the track instead of moving the bike across the screen: on the gas
        # and the whole picture keeps moving = racing (a paused race stands still, so START unpauses).
        racing = (bool(self.controls["hold"]) and now >= self.race_block_until
                  and len(self.scrolling) == self.scrolling.maxlen and np.mean(self.scrolling) >= SCROLL_SHARE
                  and any(self.last_pad[k] for k in self.controls["hold"]))
        self.race_ok = False
        if racing and not self.in_control:                   # a race, or an attract demo driving itself?
            checking = self._race_check(now)
            if checking is not None:
                self.seeking = True
                return checking
        if (self.tracker.responded and follows) or (racing and (self.in_control or self.race_ok)):
            self.in_control, self.unresponsive = True, 0.0
            self.tried.clear()
        elif was_pressing:
            self.unresponsive += dt
            self.tried.update(k for k in ("up", "down", "left", "right") if self.last_pad[k])
        # nothing followed the pad for a while, in 3+ directions (a wall blocks only some). Games that
        # steer left/right only have 2: there, both ignored for much longer - an attract demo that
        # fooled the tracker once (Galaga) - but not a crash or a map screen (a few seconds). Racing
        # games judge control by the scrolling track instead.
        one_axis = len(self.controls["axes"] or "H") == 1 and not self.controls["hold"]
        stuck = ((self.unresponsive >= NO_RESPONSE_SECONDS and len(self.tried) >= 3)
                 or (one_axis and self.unresponsive >= DEMO_SECONDS and len(self.tried) >= 2)
                 or self.still_for >= STILL_SECONDS
                 or self.lost_control)
        if stuck:
            self.in_control = False                          # judge afresh once something moves again
            self.agreement.clear()
            # ...and start the counters afresh too: left running, "stuck" held every tick and wiped
            # the evidence each time, so control could never be judged again (Double Dragon)
            self.unresponsive, self.still_for = 0.0, 0.0
            self.tried.clear()
            self.recent_follow.clear()
            self.lost_control = False
        self.seeking = not self.in_control or stuck
        if not self.seeking:
            check = self._wiggle(now)
            if check is not None:
                return check
            return self._explore(pad)
        pad = dict.fromkeys(PAD_KEYS, False)                 # nothing held: menus want clean presses
        keys = self.controls["menu"]                         # this game's menu keys (games.py)
        if self.unstick_left > 0:
            pad[keys[(self.unstick_i - 1) % len(keys)]] = True
            self.unstick_left -= 1
        elif now >= self.next_unstick and self.menu_waits < MENU_WAITS and any(
                len(q) >= 2 and np.mean(q) >= 0.5 for q in self.agreement.values()):
            # the probes seem to be moving the player: a game already running, where START
            # would pause it (Double Dragon) - keep probing and collecting evidence instead
            self.menu_waits += 1
            self.next_unstick = now + self.controls["menu_every"]
        elif now >= self.next_unstick:
            self.menu_waits = 0
            pad[keys[self.unstick_i % len(keys)]] = True
            self.unstick_i += 1
            self.unstick_left = UNSTICK_HOLD - 1
            self.next_unstick = now + self.controls["menu_every"]
            self.agreement.clear()                           # a new screen may follow: judge afresh
        elif now >= self.next_unstick - MENU_QUIET:
            pass                                             # hands off the pad before the next menu key
        else:                                                # probe for a sprite that follows the d-pad:
            self.probe_tick += 1                             # this game's directions (Excitebike changes lanes
            dirs = [k for axis in (self.controls["axes"] or "H") for k in SIDES[axis]]   # on up/down), with its
            pad[dirs[(self.probe_tick // PROBE_TICKS) % len(dirs)]] = True              # held buttons (gas) on
            for k in self.controls["hold"]:
                pad[k] = True
        return pad

    def _wiggle(self, now: float) -> dict[str, bool] | None:
        """In control, but what we track often doesn't go where we press: is it ours? Press one way,
        then the other, along the game's first axis. Ours goes both ways; an attract demo's ship,
        or an enemy in formation the tracker took for us (Galaga after game over), doesn't - then
        control is lost and seeking starts over. Racing games are left out (control there is the
        scrolling track). Returns the pad while checking."""
        if self.controls["hold"]:
            return None
        w = self.wiggle
        if w is None:
            if (len(self.recent_follow) < FOLLOW_WINDOW or np.mean(self.recent_follow) >= FOLLOW_DOUBT
                    or now < self.next_wiggle):
                return None
            w = self.wiggle = {"tick": 0, "pos": []}
        axis = (self.controls["axes"] or "H")[0]
        lo, hi = SIDES[axis]
        p = self.last_player
        w["pos"].append(None if p is None else (p["x"] if axis == "H" else p["y"]))
        w.setdefault("scroll", []).append(bool(self.scrolling) and self.scrolling[-1])
        w["tick"] += 1
        if w["tick"] <= 2 * WIGGLE_TICKS:
            pad = dict.fromkeys(PAD_KEYS, False)
            pad[lo if w["tick"] <= WIGGLE_TICKS else hi] = True
            return pad

        def moved(a, b):                                     # position change over one press
            seg = [v for v in w["pos"][a:b] if v is not None]
            return seg[-1] - seg[0] if len(seg) >= 2 else 0.0

        def scrolled(a, b):                                  # the camera followed instead (Mario mid-screen)
            return np.mean(w["scroll"][a:b]) >= 0.5

        # positions lag the press by a tick: judge each press from its second tick to one past its end
        went_lo = moved(1, WIGGLE_TICKS + 1)
        went_hi = moved(WIGGLE_TICKS + 1, 2 * WIGGLE_TICKS + 1)
        self.wiggle, self.next_wiggle = None, now + WIGGLE_EVERY
        self.recent_follow.clear()
        ok_lo = went_lo <= -WIGGLE_PX or scrolled(1, WIGGLE_TICKS + 1)
        ok_hi = went_hi >= WIGGLE_PX or scrolled(WIGGLE_TICKS + 1, 2 * WIGGLE_TICKS + 1)
        self.lost_control = not (ok_lo and ok_hi)
        return None

    def _wander_target(self) -> tuple[float, float]:
        """Where the fly heads when nothing is on screen: along the game's own axes, and in
        games with a way forward (platformers: right) mostly that way."""
        c = self.controls
        fwd = {"right": (60.0, 0.0), "left": (-60.0, 0.0), "up": (0.0, -60.0), "down": (0.0, 60.0)}
        if c.get("forward") and self.rng.random() < WANDER_FORWARD:
            return fwd[c["forward"]]
        choices = [fwd[k] for axis in (c["axes"] or "HV") for k in SIDES[axis]]
        return choices[self.rng.integers(len(choices))]

    def _explore(self, pad: dict[str, bool]) -> dict[str, bool]:
        """In control, yet nothing new for a while (no new screen, no scrolling): probably an
        overworld map, where a level is entered by stepping onto it and pressing A. Try each
        direction then A; stop as soon as a new screen appears."""
        if self.explore is None:
            if self.since_new < BORED_SECONDS:
                return pad
            self.explore = [k for k, n in EXPLORE_STEPS for _ in range(n)]
        if self.since_new == 0.0 or not self.explore:          # a new screen: it worked (or we're done)
            self.explore = None
            self.since_new = 0.0 if self.since_new == 0.0 else self.since_new - BORED_SECONDS / 2
            return pad
        key = self.explore.pop(0)
        out = dict.fromkeys(PAD_KEYS, False)
        if key:
            out[key] = True
        return out

    def _vault(self, pad: dict[str, bool], player: dict | None, dt: float) -> dict[str, bool]:
        """Platformers: the fly pushes forward but gets nowhere (a pipe, a wall, a step) and the
        screen isn't scrolling either. A tap of jump is a hop; clearing it takes the jump button
        held down while still going forward (SMB jumps higher the longer A is held)."""
        c = self.controls
        fwd, jump = c.get("forward"), c["jump"]
        if not fwd or not jump:
            return pad
        hold, rest = VAULT_TICKS
        if self.vault_left > 0:
            self.vault_left -= 1
            out = dict(pad, **{fwd: True, jump: self.vault_left >= rest})
            out[OPPOSITE[fwd]] = False
            return out
        if not pad[fwd] or player is None:
            self.push = None
            return pad
        axis, sign = {"right": ("x", 1), "left": ("x", -1), "down": ("y", 1), "up": ("y", -1)}[fwd]
        scrolled = len(self.scrolling) >= 10 and np.mean(list(self.scrolling)[-10:]) >= 0.5
        if self.push is None or scrolled or sign * (player[axis] - self.push[0]) > VAULT_PROGRESS:
            self.push = (player[axis], 0.0)                   # getting somewhere: start counting afresh
            return pad
        self.push = (self.push[0], self.push[1] + dt)
        if self.push[1] >= VAULT_SECONDS:
            self.push, self.vault_left = None, hold + rest
        return pad

    def _race_check(self, now: float) -> dict[str, bool] | None:
        """The screen scrolls while we are on the gas: a race, or an attract demo driving itself
        (Rad Racer)? Press START once and watch: a demo gives way to another screen; a race
        pauses (stands still), so press START again to resume it. Returns the pad while checking."""
        pad = dict.fromkeys(PAD_KEYS, False)
        rc = self.race_check
        if rc is None:
            rc = self.race_check = {"stage": "quiet", "t": now, "changes": []}
        if rc["stage"] == "quiet":                           # hands off first (see MENU_QUIET)
            if now - rc["t"] >= MENU_QUIET:
                rc.update(stage="press", t=now)
            return pad
        if rc["stage"] == "press":
            pad["start"] = True
            if now - rc["t"] >= 0.1:
                rc.update(stage="watch", t=now)
            return pad
        if rc["stage"] == "watch":
            rc["changes"].append(self.last_change)
            if now - rc["t"] < 1.5:
                return pad
            ch = np.array(rc["changes"])
            if ch.max() > SCENE_CUT:                         # a demo: START took us to another screen
                self.race_check = None
                self.race_block_until = now + 15.0
                self.scrolling.clear()
                return None
            if (ch > SCROLL_LEVEL).mean() < 0.3:             # it stood still: a real race, now paused
                rc.update(stage="resume", t=now)
                return pad
            self.race_check, self.race_ok = None, True       # START changed nothing and it still moves
            return None
        if now - rc["t"] < MENU_QUIET:                       # resume the paused race
            return pad
        pad["start"] = True
        if now - rc["t"] >= MENU_QUIET + 0.1:
            self.race_check, self.race_ok = None, True
        return pad

    def idle_tick(self) -> None:
        flies = self._step([])
        self.decoder.observe(flies)
        self.frame_spikes = flies[0]


def parse_encoder(text: str) -> dict:
    out = {}
    for item in filter(None, (text or "").split(",")):
        name, value = item.split("=", 1)
        out[name.strip()] = float(value)
    return out


NO_GAME = "_no_game"                 # name the fly runs under until a game is loaded


def first_rom(folder: Path) -> Path | None:
    """The first NES game in the ROM folder, if there is one."""
    if not folder.is_dir():
        return None
    return next((p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in (".nes", ".zip")
                 and nes_inside(p)), None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dash-port", type=int, default=8777)
    p.add_argument("--rom", default=None, help="game to start with (default: the first game in the ROM folder)")
    p.add_argument("--rom-dir", default=None,
                   help="folder of your ROMs for the dashboard's game list (default: the --rom folder, else "
                        "$FLY_ROMS, else roms/ in this repo)")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--voters", type=int, default=None,
                   help="flies per axis that vote (default 4 on GPU, 1 on CPU)")
    p.add_argument("--seed", type=int, default=64)
    p.add_argument("--encoder", default="loom_size=0.6", help="e.g. loom_size=0.6,chase_gain=0.3")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--checkpoint", default=None,
                   help="learning checkpoint (default mesen/checkpoints/<rom>.npz); resumes if it exists")
    p.add_argument("--no-learn", action="store_true", help="play with the brain alone, no learning")
    p.add_argument("--engine", choices=["server", "browser"], default="server",
                   help="server: the game runs in this process and keeps running whatever the browser does "
                        "(default); browser: the game runs in the dashboard page (EmulatorJS)")
    p.add_argument("--learner", choices=["mc", "td"], default="mc",
                   help="mc: credit each choice with the next second's reward (default). td: longer horizon "
                        "(n-step TD), forgetting and object positions - not better in tests so far; own checkpoint")
    p.add_argument("--motion", choices=["blobs", "columns"], default="blobs",
                   help="how pixels reach the motion detectors: blobs (things growing -> T4a/T5a) or columns "
                        "(every T4/T5 cell by its own column and preferred direction; learns in its own checkpoint)")
    p.add_argument("--vision", choices=["both", "eyes", "sprites"], default="both",
                   help="eyes: screen pixels -> motion detectors T4/T5; sprites: object detectors LC10a/LPLC2/LC4/LPLC1")
    args = p.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if cuda_available() else "cpu"
    voters = args.voters or (4 if device == "cuda" else 1)
    rom_dir = Path(args.rom_dir) if args.rom_dir else (Path(args.rom).parent if args.rom else GAME_DIR)
    rom = Path(args.rom) if args.rom else first_rom(rom_dir)
    name = rom.stem if rom else NO_GAME
    checkpoint = None if args.no_learn else Path(args.checkpoint or HERE / "checkpoints" /
                                                 checkpoint_name(name, args.motion, args.learner))
    fly = FlyNes(device=device, voters=voters, seed=args.seed, encoder=parse_encoder(args.encoder),
                 vision=args.vision, checkpoint=checkpoint, rom_name=name, motion=args.motion,
                 learner=args.learner)
    engine = args.engine
    if engine == "server":
        try:
            from server_emu import ServerEmulator
        except ImportError:
            print("cynes is not installed (pip install cynes): the game runs in the browser instead", flush=True)
            engine = "browser"
    dash = Dashboard(fly, port=args.dash_port, rom=rom, rom_dir=rom_dir, engine=engine)
    dash.start(open_browser=not args.no_browser)
    if rom is None:
        print(f"no game yet: put NES ROMs you own (.nes, or .zip holding one) in {rom_dir}, "
              "or use '+ Add game' on the dashboard", flush=True)
    elif engine == "server":
        dash.emu = ServerEmulator(dash, rom)
        print(f"game running in the server: {rom.name}", flush=True)
    try:
        while True:
            time.sleep(3600)
    finally:
        fly.close()

if __name__ == "__main__":
    main()
