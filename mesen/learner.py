"""Learning on top of the frozen fly brain, with checkpoints.

The connectome never changes. What learns is a linear readout of its descending
neurons (reservoir computing, as in sshfighter/reservoir.py): for each choice the
pad can make, an estimate of how much reward follows when the brain is in this
state and that choice is taken.

    heads   H: left / none / right    V: up / none / down    fire: <fire button> / none
            (only the axes and buttons the game uses, see games.py)
    reward  score going up (+1 per typical gain), losing a life (-10), read from game RAM;
            where score and lives live is found automatically (RewardFinder)
    target  the reward collected over the next second (Monte Carlo, no bootstrapping)
    fit     ridge regression, kept as running sums (X'X, X'y) so it trains online
            and a checkpoint is just those sums

The brain's own choice stays the default. The learner overrides it when one choice
is estimated clearly better, and explores (a random choice held for half a second)
a fraction of the time, which is where most of what it learns comes from.
"""
from __future__ import annotations

import json
import re
import time
from collections import deque
from pathlib import Path

import numpy as np

HORIZON = 20                 # ticks of reward credited to a choice directly (20 x 50 ms = 1 s)...
GAMMA = 0.985                # ...discounted per tick (credit half-life ~2.3 s), and beyond the second the
                             # learner's own estimate of what follows (n-step TD): chains longer than a
                             # second ("line up, shoot, the enemy dies, safety") get credit back to their start
FORGET = 1 - 1 / 30000       # per sample: old samples fade (their bootstrapped targets go stale; games change)
V_CLIP = 100.0               # bound on the bootstrapped estimate
LEARNER_VERSION = 2          # 1: 1 s Monte Carlo, brain features only (default). 2: n-step TD + forgetting +
                             # object features (--learner td; tested, not better than 1 so far)
EXPLORE_START, EXPLORE_END = 0.2, 0.05
EXPLORE_HALF_LIFE = 20000    # samples until exploration has come half way down
EXPLORE_HOLD = 10            # ticks an exploratory choice is held
MIN_SAMPLES = 300            # per choice, before its estimate is trusted
MARGIN = 0.05                # how much better (reward units) a choice must look to override the brain
RIDGE = 10.0
SAVE_EVERY = 60.0            # seconds

# Known maps (found by watching RAM while playing). Everything else is found by RewardFinder.
#   score: n bytes from addr, "digits" = one decimal digit per byte, "bcd" = two per byte
#   progress: where the player is in the level (page, x). New ground is rewarded too - otherwise the
#   enemies ahead teach a platformer fly that standing still is safest (running out of time costs a
#   life, but long after the second).
PROFILES = {
    r"galaga": {"score": {"addr": 0xE0, "n": 6, "fmt": "digits", "order": "big", "unit": 100.0}, "lives": 0x485},
    # player score, not TOP at $07D7: that one never resets, so the finder prefers it, but it stops
    # rising once the old record is beaten back; not SMB 2/3 either (other RAM)
    r"super mario bros(?!\.?\s*[23])": {"score": {"addr": 0x7DD, "n": 6, "fmt": "digits", "order": "big",
                                                  "unit": 10.0}, "lives": 0x75A,
                                        "progress": {"hi": 0x6D, "lo": 0x86, "px": 160.0}},
}


def heads_for(controls: dict | None) -> dict[str, tuple]:
    c = controls or {"axes": "HV", "fire": "a"}
    heads = {}
    if "H" in c["axes"]:
        heads["H"] = ("left", None, "right")
    if "V" in c["axes"]:
        heads["V"] = ("up", None, "down")
    if c.get("fire"):
        heads["fire"] = (c["fire"], None)
    return heads


def profile_for(rom_name: str) -> dict | None:
    name = rom_name.lower()
    return next((p for key, p in PROFILES.items() if re.search(key, name)), None)


def read_score(ram, spec: dict) -> int | None:
    b = [int(ram[spec["addr"] + i]) for i in range(spec["n"])]
    if spec["order"] == "little":
        b = b[::-1]
    if spec["fmt"] == "digits":
        if any(v > 9 for v in b):
            return None
        return int("".join(map(str, b)))
    nibbles = [(v >> 4, v & 15) for v in b]
    if any(h > 9 or lo > 9 for h, lo in nibbles):
        return None
    return int("".join(f"{h}{lo}" for h, lo in nibbles))


class Reward:
    def __init__(self, profile: dict):
        self.p = profile
        self.score = None
        self.lives = None
        self.best = None                     # furthest point reached in the level this life

    def __call__(self, ram: np.ndarray | None) -> float:
        if ram is None or len(ram) < 0x800:
            return 0.0
        score = read_score(ram, self.p["score"]) if self.p.get("score") else None
        lives = int(ram[self.p["lives"]]) if self.p.get("lives") is not None else None
        r = 0.0
        if score is not None and self.score is not None and 0 < score - self.score < 50000:
            r += (score - self.score) / self.p["score"].get("unit", 100.0)
        if lives is not None and self.lives is not None and lives == self.lives - 1:
            r -= 10.0
        prog = self.p.get("progress")
        if prog:
            pos = int(ram[prog["hi"]]) * 256 + int(ram[prog["lo"]])
            if self.best is None or pos < self.best - 512:   # a new life or level: start counting again
                self.best = pos
            elif pos > self.best:
                r += (pos - self.best) / prog["px"]
                self.best = pos
        self.score = score if score is not None else self.score
        self.lives = lives
        return r


class RewardFinder:
    """Watch RAM while the fly plays and find where the game keeps score and lives.

    score: a run of 3-7 bytes that reads as a decimal number (one digit per byte, or
    packed BCD, either byte order) that only goes up, except back to about zero when a
    new game starts; that goes up in steps of different sizes (points differ per enemy);
    that is not a clock (rising in most seconds); and that rises while the fly is in
    control, not during title screens and attract demos (which keep their own). lives: a small counter that drops
    by exactly one, where the drops happen when the player's sprite vanishes (the fly
    loses its ship / character)."""

    EVERY = 20               # ticks between snapshots (1 s)
    MIN_SNAPSHOTS = 120      # 2 minutes of play before trying
    NEAR = 4                 # snapshots: a drop in lives within this of losing the player counts

    def __init__(self):
        self.snaps: list[np.ndarray] = []
        self.lost: list[bool] = []
        self.playing: list[bool] = []
        self.lost_now = False
        self.playing_now = False
        self.tick = 0
        self.found: dict | None = None
        self.tried_at = 0

    def observe(self, ram: np.ndarray | None, player_lost: bool = False, playing: bool = True) -> dict | None:
        self.tick += 1
        self.lost_now |= player_lost
        self.playing_now |= playing
        if ram is None or len(ram) < 0x800 or self.tick % self.EVERY:
            return None
        self.snaps.append(ram[:0x800].copy())
        self.lost.append(self.lost_now)
        self.playing.append(self.playing_now)
        self.lost_now = self.playing_now = False
        if len(self.snaps) > 3600:
            self.snaps, self.lost, self.playing = self.snaps[-3600:], self.lost[-3600:], self.playing[-3600:]
        n = len(self.snaps)
        if n >= self.MIN_SNAPSHOTS and n - self.tried_at >= 30:
            self.tried_at = n
            self.found = self.analyse(np.array(self.snaps, np.int32), np.array(self.lost), np.array(self.playing))
        return self.found

    @staticmethod
    def _score(R: np.ndarray, playing: np.ndarray | None = None) -> dict | None:
        T = len(R)
        playing = np.ones(T, bool) if playing is None else playing
        best, best_key = None, None
        for n in range(3, 8):
            for addr in range(0, 0x800 - n):
                W = R[:, addr:addr + n]
                if W.max() == W.min():
                    continue
                for fmt in ("digits", "bcd"):
                    if fmt == "digits" and W.max() > 9:
                        continue
                    if fmt == "bcd" and (((W >> 4) > 9).any() or ((W & 15) > 9).any()):
                        continue
                    for order in ("big", "little"):
                        V = W if order == "big" else W[:, ::-1]
                        if fmt == "digits":
                            val = (V * (10 ** np.arange(n - 1, -1, -1))).sum(axis=1)
                        else:
                            dig = np.stack([V >> 4, V & 15], axis=2).reshape(T, 2 * n)
                            val = (dig * (10 ** np.arange(2 * n - 1, -1, -1))).sum(axis=1)
                        d = np.diff(val)
                        up = d[d > 0]
                        downs = d < 0
                        if len(up) < 5 or len(up) > 0.5 * T or downs.sum() > 3:
                            continue                         # too few changes, or a clock
                        if not np.all(val[1:][downs] <= 0.05 * np.maximum(val[:-1][downs], 1)):
                            continue                         # only new games may bring it back down
                        if len(np.unique(up)) < 2:
                            continue                         # always the same step: a counter, not points
                        if playing[1:][d > 0].mean() < 0.9:
                            continue                         # rises in demos / title screens too
                        key = (n >= 4, -int(downs.sum()), len(up), n)   # real scores reset only per game
                        if best_key is None or key > best_key:
                            best_key, best = key, {"addr": addr, "n": n, "fmt": fmt, "order": order}
        while best and best["n"] > 5:                        # drop high-order bytes that never changed
            hi = best["addr"] if best["order"] == "big" else best["addr"] + best["n"] - 1
            if R[:, hi].max() != R[:, hi].min():
                break
            best["n"] -= 1
            if best["order"] == "big":
                best["addr"] += 1
        return best

    @staticmethod
    def score_values(R: np.ndarray, spec: dict) -> np.ndarray:
        W = R[:, spec["addr"]:spec["addr"] + spec["n"]]
        V = W if spec["order"] == "big" else W[:, ::-1]
        if spec["fmt"] == "bcd":
            V = np.stack([V >> 4, V & 15], axis=2).reshape(len(R), -1)
        return (V * (10 ** np.arange(V.shape[1] - 1, -1, -1))).sum(axis=1)

    @staticmethod
    def _lives(R: np.ndarray, score: np.ndarray) -> int | None:
        """A small counter that only drops by one, and only jumps back up (to its maximum)
        when the score goes back to zero: a new game. The one that drops most often wins."""
        best, best_drops = None, 0
        for addr in range(0x800):
            x = R[:, addr]
            if x.max() > 9 or x.max() < 2:
                continue
            d = np.diff(x)
            drops, ups = np.flatnonzero(d == -1), np.flatnonzero(d > 0)
            if len(drops) < 2 or (d < -1).any() or not np.all(x[1:][ups] == x.max()):
                continue
            if np.all(score[ups + 1] <= 0.05 * np.maximum(score[ups], 1)) and len(drops) > best_drops:
                best, best_drops = addr, len(drops)
        return best

    def analyse(self, R: np.ndarray, lost: np.ndarray, playing: np.ndarray) -> dict | None:
        score = self._score(R, playing)
        if score is None:
            return None
        val = self.score_values(R, score)
        steps = np.diff(val)
        score["unit"] = float(np.median(steps[steps > 0]))      # a typical gain; reward = gain / unit
        return {"score": score, "lives": self._lives(R, val), "found": True}


class Learner:
    def __init__(self, dim: int, path: Path, rom_name: str, heads: dict[str, tuple], td: bool = False):
        self.td = td                                     # False: version 1 (1 s Monte Carlo, no forgetting)
        self.path = path
        self.rom = rom_name
        self.dim = dim
        self.heads = heads
        self.XtX = {h: np.zeros((len(o), dim, dim)) for h, o in heads.items()}
        self.Xty = {h: np.zeros((len(o), dim)) for h, o in heads.items()}
        self.n = {h: np.zeros(len(o), np.int64) for h, o in heads.items()}
        self.w = {h: np.zeros((len(o), dim)) for h, o in heads.items()}
        self.pending: deque = deque()
        self.rng = np.random.default_rng()
        self.explore = {h: (None, 0) for h in heads}     # (choice index, ticks left)
        self.total_reward = 0.0
        self.samples = 0
        self.overrides = 0
        self.ticks = 0
        self.last_save = time.monotonic()
        self.recent = deque(maxlen=1200)                 # rewards over the last minute of ticks
        self.extra: dict = {}
        self.profile: dict | None = None

    # ---------------------------------------------------------------- checkpoints
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"{k}_{h}": getattr(self, k)[h] for k in ("XtX", "Xty", "n") for h in self.heads}
        arrays.update({f"extra_{k}": v for k, v in self.extra.items()})
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez_compressed(tmp, dim=self.dim, rom=self.rom, samples=self.samples,
                            version=LEARNER_VERSION if self.td else 1,
                            total_reward=self.total_reward, heads=json.dumps(self.heads),
                            profile=json.dumps(self.profile), **arrays)
        tmp.replace(self.path)
        self.last_save = time.monotonic()

    def load(self) -> bool:
        if not self.path.is_file():
            return False
        z = np.load(self.path, allow_pickle=False)
        if "XtX_H" not in z.files and "XtX_fire" not in z.files and "XtX_V" not in z.files:
            return False                                   # calibration / profile only, nothing learned yet
        version = int(z["version"]) if "version" in z.files else 1
        if version != (LEARNER_VERSION if self.td else 1):
            # learned with another learner: those sums mean something else. Keep the file, start fresh.
            z.close()
            old = self.path.with_name(f"{self.path.stem}.v{version}.npz")
            self.path.replace(old)
            print(f"learning: {self.path.name} was made by learner version {version}; kept as {old.name}, "
                  f"starting fresh with version {LEARNER_VERSION}", flush=True)
            return False
        if int(z["dim"]) != self.dim:
            print(f"checkpoint {self.path.name} has {int(z['dim'])} features, not {self.dim}: starting fresh", flush=True)
            return False
        for h, opts in self.heads.items():
            if f"XtX_{h}" in z.files and z[f"XtX_{h}"].shape[0] == len(opts):
                self.XtX[h], self.Xty[h], self.n[h] = z[f"XtX_{h}"], z[f"Xty_{h}"], z[f"n_{h}"]
                self.refit(h)
        self.samples, self.total_reward = int(z["samples"]), float(z["total_reward"])
        return True

    @staticmethod
    def saved_profile(path: Path | None) -> dict | None:
        if path is None or not path.is_file():
            return None
        z = np.load(path, allow_pickle=False)
        return json.loads(str(z["profile"])) if "profile" in z.files else None

    # ---------------------------------------------------------------- learning
    def refit(self, head: str) -> None:
        for i in range(len(self.heads[head])):
            A = self.XtX[head][i] + RIDGE * np.eye(self.dim)
            self.w[head][i] = np.linalg.solve(A, self.Xty[head][i])

    def epsilon(self) -> float:
        k = 0.5 ** (self.samples / EXPLORE_HALF_LIFE)
        return EXPLORE_END + (EXPLORE_START - EXPLORE_END) * k

    def step(self, feats: dict[str, np.ndarray], brain_pad: dict[str, bool], reward: float) -> dict[str, bool]:
        """feats: one feature vector per head; reward: what the game paid since last tick."""
        self.ticks += 1
        self.total_reward += reward
        self.recent.append(reward)
        g = GAMMA if self.td else 1.0
        for item in self.pending:                           # credit this tick's reward to recent choices
            item[2] += reward * g ** (self.ticks - item[3] - 1)
        while self.pending and self.ticks - self.pending[0][3] >= HORIZON:
            head, x, G, _, i = self.pending.popleft()
            if self.td and self.n[head].sum() >= MIN_SAMPLES:      # (the head as a whole has some experience)
                # n-step TD: what the learner now expects from here on, from this tick's state
                G += g ** HORIZON * float(np.clip(np.max(self.w[head] @ feats[head]), -V_CLIP, V_CLIP))
                self.XtX[head][i] *= FORGET
                self.Xty[head][i] *= FORGET
            self.XtX[head][i] += np.outer(x, x)
            self.Xty[head][i] += x * G
            self.n[head][i] += 1
            self.samples += 1
            if self.samples % 200 == 0:
                for h in self.heads:
                    self.refit(h)
        pad = dict(brain_pad)
        for head, options in self.heads.items():
            x = feats[head]
            brain_i = next((i for i, o in enumerate(options) if o and brain_pad.get(o)), options.index(None))
            choice, left = self.explore[head]
            if left <= 0 and self.rng.random() < self.epsilon():
                choice, left = int(self.rng.integers(len(options))), EXPLORE_HOLD
            if left > 0:
                self.explore[head] = (choice, left - 1)
                i = choice
            else:
                self.explore[head] = (None, 0)
                i = brain_i
                if self.n[head].min() >= MIN_SAMPLES:
                    q = self.w[head] @ x
                    best = int(np.argmax(q))
                    if best != brain_i and q[best] - q[brain_i] > MARGIN:
                        i = best
                        self.overrides += 1
            for o in options:
                if o:
                    pad[o] = False
            if options[i]:
                pad[options[i]] = True
            self.pending.append([head, x, 0.0, self.ticks, i])
        if time.monotonic() - self.last_save > SAVE_EVERY:
            self.save()
        return pad

    def relabel(self, pad: dict[str, bool]) -> None:
        """The pad that really went out (a person on the dashboard pad can change it): the choices
        just queued by step() are rewritten to match, so the reward is credited to what was pressed."""
        recent = list(self.pending)[-len(self.heads):]
        for item in recent:
            options = self.heads[item[0]]
            item[4] = next((i for i, o in enumerate(options) if o and pad.get(o)), options.index(None))

    def status(self) -> dict:
        return {"samples": int(self.samples), "explore": round(self.epsilon(), 3),
                "reward_per_min": round(float(sum(self.recent)) * 1200 / max(len(self.recent), 1), 1),
                "total_reward": round(self.total_reward, 1), "overrides": int(self.overrides),
                "ready": {h: bool(self.n[h].min() >= MIN_SAMPLES) for h in self.heads},
                "checkpoint": self.path.name, "version": LEARNER_VERSION if self.td else 1}
