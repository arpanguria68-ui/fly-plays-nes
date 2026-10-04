"""The fly's reward feed: game events, with the meaning they have for an animal, turned into the
reward the learner learns from - and, optionally, into taste and dopamine for the brain itself.

Events are detected the same way in every game, from what the fly code already tracks (the score
and lives in RAM, the objects on screen, the player) - no per-game code:

    food      the score went up                         like eating: the basic reward
    kill      an enemy vanished as the score went up    a threat eliminated
    relief    a close threat is gone                    safety (scaled by how close it was)
    consume   the player touched something, it vanished picking up an item
    explore   new ground reached (games with a known level position)
    danger    an enemy is close (per second)            fear
    pain      a life was lost
    survive   staying alive (per second)
    + RAM rules: "this byte goes up / down / changes" (coins, health, keys...)

Each event has a weight (0 turns it off, negative punishes). Settings live per game in
mesen/rewards/<game>.json, start from the genre's defaults, and are edited live from the
dashboard (/rewards).

Taste (off by default): positive rewards also excite sweet-taste neurons (LB3, LB2d) and reward
dopamine neurons (PAM), negative ones bitter-taste neurons (LB1) and punishment dopamine (PPL1),
as food and pain do in a real fly. The connectome here has no plasticity, so this changes what
the brain does in the moment, not its wiring; the learning itself is the learner's.
"""
from __future__ import annotations

import json
import re
import time
from collections import deque
from pathlib import Path

import numpy as np

from learner import read_score

HERE = Path(__file__).resolve().parent
RULES_DIR = HERE / "rewards"

EVENTS = {
    "food": {"label": "Food: the score went up", "unit": "per typical score gain", "default": 1.0},
    "kill": {"label": "Kill: an enemy eliminated", "unit": "per enemy", "default": 2.0},
    "relief": {"label": "Relief: a close threat is gone", "unit": "x closeness", "default": 1.0},
    "consume": {"label": "Consume: touched an item, it vanished", "unit": "per item", "default": 1.0},
    "explore": {"label": "Explore: new ground reached", "unit": "per 160 px", "default": 1.0},
    "danger": {"label": "Danger: an enemy is close", "unit": "per second", "default": -0.5},
    "pain": {"label": "Pain: a life lost", "unit": "per life", "default": -10.0},
    "survive": {"label": "Survive: staying alive", "unit": "per second", "default": 0.0},
}
# How strongly each sense drives the brain (x the calibrated level). The readouts were calibrated
# at 1x, so above 1 the fly reacts to that sense more readily, below 1 it pays it less attention.
STIMULUS = {
    "chase": "Chase: targets (LC10a)", "loom": "Looming: things growing (LPLC2)",
    "threat": "Threat: close attackers (LC4)", "shot": "Shots: small fast things (LPLC1)",
    "eyes": "Motion eyes: the picture (T4/T5)", "taste": "Taste strength (sweet/bitter, PAM/PPL1)",
}
STIM_MAX = 3.0
GENRE_DEFAULTS = {
    "shooter_v": {"kill": 2.0, "danger": -0.5}, "shooter_h": {"kill": 2.0, "danger": -0.5},
    "run_gun": {"kill": 2.0, "danger": -0.5, "explore": 1.0},
    "platformer": {"kill": 1.0, "consume": 1.0, "danger": -0.3, "explore": 1.0},
    "beat_em_up": {"kill": 2.0, "danger": -0.3}, "fighting": {"kill": 2.0, "danger": 0.0},
    "racing": {"danger": 0.0, "kill": 0.0, "survive": 0.2}, "sports": {"danger": 0.0, "kill": 0.0},
    "puzzle": {"danger": 0.0, "kill": 0.0, "consume": 0.0}, "light_gun": {"danger": 0.0},
    "maze": {"consume": 1.5, "danger": -0.5}, "topdown": {"kill": 2.0, "danger": -0.5},
}
THREAT_PX = 70.0            # an enemy closer than this is a danger (closeness 0..1)
TOUCH_PX = 16.0             # an object vanishing this close to the player was touched (consumed)
EDGE_PX = 12.0              # vanishing this close to the screen edge is leaving, not dying
MATCH_TICKS = 15            # a vanish and a score gain this close in time make a kill (0.75 s)
CONFIRM_TICKS = 10          # a touch is a pickup if no life is lost this soon after
ITEM_AGE = 20               # ticks an object must have been on screen to count as a pickup (1 s)...
ENEMY_AGE = 10              # ...or as a kill (0.5 s): pieces of the player's own sprite come and go faster
LOG_SIZE = 40
TASTE_TAU = 0.3             # s: how long a taste lingers
TASTE_MAX = 0.6             # voltage per step at full taste


def rules_path(rom_name: str) -> Path:
    return RULES_DIR / (re.sub(r"[^\w().,+ -]", "_", rom_name).strip() + ".json")


def default_config(genre: str) -> dict:
    w = {k: v["default"] for k, v in EVENTS.items()}
    w.update(GENRE_DEFAULTS.get(genre, {}))
    return {"weights": w, "taste": False, "ram_rules": [], "stimulus": dict.fromkeys(STIMULUS, 1.0)}


def load_config(rom_name: str, genre: str) -> dict:
    cfg = default_config(genre)
    p = rules_path(rom_name)
    if p.is_file():
        try:
            saved = json.loads(p.read_text(encoding="utf-8"))
            cfg["weights"].update({k: float(v) for k, v in saved.get("weights", {}).items() if k in EVENTS})
            cfg["taste"] = bool(saved.get("taste", False))
            cfg["ram_rules"] = clean_rules(saved.get("ram_rules", []))
            cfg["stimulus"].update(clean_stimulus(saved.get("stimulus", {})))
        except (ValueError, OSError):
            pass
    return cfg


def clean_rules(rules) -> list[dict]:
    """RAM rules from the user: name, addr (0x000-0x7FF), when (up/down/change), weight."""
    out = []
    for r in rules or []:
        try:
            addr = int(str(r.get("addr", "")), 0)
        except ValueError:
            continue
        if not 0 <= addr < 0x800 or r.get("when") not in ("up", "down", "change"):
            continue
        out.append({"name": str(r.get("name") or f"RAM {addr:#05x}")[:40], "addr": addr,
                    "when": r["when"], "weight": float(r.get("weight", 1.0))})
    return out[:16]


def clean_stimulus(gains) -> dict:
    return {k: max(0.0, min(STIM_MAX, float(v))) for k, v in (gains or {}).items() if k in STIMULUS}


def save_config(rom_name: str, cfg: dict) -> None:
    RULES_DIR.mkdir(exist_ok=True)
    data = {"weights": cfg["weights"], "taste": cfg["taste"], "stimulus": cfg["stimulus"],
            "ram_rules": [{**r, "addr": f"{r['addr']:#05x}"} for r in cfg["ram_rules"]]}
    rules_path(rom_name).write_text(json.dumps(data, indent=2), encoding="utf-8")


class RewardFeed:
    """Called once per tick with the RAM and the fly; returns that tick's reward."""

    def __init__(self, profile: dict, controls: dict, rom_name: str, brain=None):
        self.p = profile
        self.rom_name, self.genre = rom_name, controls.get("genre", "")
        self.cfg = load_config(rom_name, self.genre)
        self.score = self.lives = self.best = None
        self.ram_prev: np.ndarray | None = None
        self.prev_tracks: dict[int, dict] = {}
        self.pending: deque = deque()            # vanished objects waiting to be judged
        self.tick_no = 0
        self.pain_tick = -10**9
        self.rises: deque = deque()
        self.log: deque = deque(maxlen=LOG_SIZE)
        self.recent: deque = deque()             # (time, event, reward) for the last minute
        self.taste_pos = self.taste_neg = 0.0
        self.ticks_seen: deque = deque()         # (time, playing, danger on) for the last minute
        self.play_ticks = self.score_gains = 0   # since this game started
        self.scout = RamScout(profile)
        self.cells = {}
        if brain is not None:
            ct = brain.cell_type.astype(str)
            pick = lambda pat: np.flatnonzero(np.array([bool(re.match(pat, t)) for t in ct]))
            self.cells = {"sweet": pick(r"LB3$|LB2d$"), "pam": pick(r"PAM\d"),
                          "bitter": pick(r"LB1"), "ppl1": pick(r"PPL1\d")}

    # ---------------------------------------------------------------- settings
    def set_config(self, cfg: dict) -> None:
        self.cfg = cfg

    def weight(self, ev: str) -> float:
        return float(self.cfg["weights"].get(ev, 0.0))

    # ---------------------------------------------------------------- one tick
    def __call__(self, ram: np.ndarray | None, fly, dt: float = 0.05) -> float:
        self.tick_no += 1
        events: list[tuple[str, float, str]] = []      # (event, value before weight, note)
        playing = not fly.seeking
        if ram is not None and len(ram) >= 0x800:
            self._ram_events(ram, events, playing)
            self.scout.observe(ram, playing, any(e[0] == "pain" for e in events),
                               any(e[0] == "food" for e in events))
        if playing:
            self._object_events(fly, events, dt)
        else:
            self.prev_tracks, self.pending = {}, deque()
        total = 0.0
        now = time.monotonic()
        for ev, value, note in events:
            w = self.weight(ev) if ev in EVENTS else value       # RAM rules carry their own weight
            r = w * (value if ev in EVENTS else 1.0)
            if r == 0:
                continue
            total += r
            self.recent.append((now, ev, r))
            if ev not in ("danger", "survive", "explore") or abs(r) >= 0.5:    # (small steady ones would flood the log)
                self.log.appendleft({"t": round(now, 2), "event": ev, "reward": round(r, 2), "note": note})
        while self.recent and now - self.recent[0][0] > 60:
            self.recent.popleft()
        self.ticks_seen.append((now, playing, playing and any(e[0] == "danger" for e in events)))
        self.play_ticks += playing
        self.score_gains += any(e[0] == "food" for e in events)
        while self.ticks_seen and now - self.ticks_seen[0][0] > 60:
            self.ticks_seen.popleft()
        decay = np.exp(-dt / TASTE_TAU)
        self.taste_pos = self.taste_pos * decay + max(total, 0.0)
        self.taste_neg = self.taste_neg * decay + max(-total, 0.0)
        return total

    def _ram_events(self, ram, events, playing) -> None:
        p = self.p
        score = read_score(ram, p["score"]) if p.get("score") else None
        lives = int(ram[p["lives"]]) if p.get("lives") is not None else None
        if score is not None and self.score is not None and 0 < score - self.score < 50000 and playing:
            gain = score - self.score
            events.append(("food", gain / p["score"].get("unit", 100.0), f"score +{gain}"))
            self.rises.append(self.tick_no)               # each score gain can pay for one kill
        if lives is not None and self.lives is not None and lives == self.lives - 1:
            events.append(("pain", 1.0, "life lost"))
            self.pain_tick = self.tick_no                # touches around now killed us: not pickups
        prog = p.get("progress")
        if prog and playing:
            pos = int(ram[prog["hi"]]) * 256 + int(ram[prog["lo"]])
            if self.best is None or pos < self.best - 512:
                self.best = pos
            elif pos > self.best:
                events.append(("explore", (pos - self.best) / prog.get("px", 160.0), f"+{pos - self.best} px"))
                self.best = pos
        if self.ram_prev is not None and playing:
            for rule in self.cfg["ram_rules"]:
                d = int(ram[rule["addr"]]) - int(self.ram_prev[rule["addr"]])
                hit = (d > 0) if rule["when"] == "up" else (d < 0) if rule["when"] == "down" else (d != 0)
                if hit:
                    events.append((rule["name"], rule["weight"], f"{rule['addr']:#05x} {d:+d}"))
        self.score = score if score is not None else self.score
        self.lives = lives
        self.ram_prev = np.array(ram[:0x800], np.uint8)

    def _object_events(self, fly, events, dt) -> None:
        player = fly.last_player
        tracks = {t["id"]: t for t in fly.tracker.tracks}
        pid = fly.tracker.player_id
        # danger: the closest enemy
        if player is not None:
            close = [max(0.0, 1 - np.hypot(t["x"] - player["x"], t["y"] - player["y"]) / THREAT_PX)
                     for k, t in tracks.items() if k != pid and t.get("still", 0) < 60]
            c = max(close, default=0.0)
            if c > 0:
                events.append(("danger", c * dt, f"closeness {c:.2f}"))
            events.append(("survive", dt, ""))
        # objects that vanished mid-screen since last tick: killed, or picked up
        for k, t in self.prev_tracks.items():
            if k in tracks or k == pid or player is None:
                continue
            x, y = t["x"], t["y"]
            if x < EDGE_PX or x > 256 - EDGE_PX or y < EDGE_PX or y > 240 - EDGE_PX:
                continue
            d = float(np.hypot(x - player["x"], y - player["y"]))
            if t.get("age", 0) < (ITEM_AGE if d < TOUCH_PX else ENEMY_AGE):
                continue                                     # a flicker or a piece of the player, not a thing
            self.pending.append({"tick": self.tick_no, "touch": d < TOUCH_PX,
                                 "close": max(0.0, 1 - d / THREAT_PX), "where": (round(x), round(y))})
        while self.rises and self.tick_no - self.rises[0] > 2 * MATCH_TICKS:
            self.rises.popleft()
        keep = deque()
        for v in self.pending:
            age = self.tick_no - v["tick"]
            if v["touch"]:
                if self.pain_tick >= v["tick"] - 1:
                    continue                                 # touched it and died: an enemy, not food
                if age >= CONFIRM_TICKS:
                    events.append(("consume", 1.0, f"item at {v['where']}"))
                else:
                    keep.append(v)
            elif any(abs(t - v["tick"]) <= MATCH_TICKS for t in self.rises):
                self.rises.remove(next(t for t in self.rises if abs(t - v["tick"]) <= MATCH_TICKS))
                events.append(("kill", 1.0, f"enemy at {v['where']}"))
                if v["close"] > 0:
                    events.append(("relief", v["close"], f"it was {v['close']:.2f} close"))
            elif age < MATCH_TICKS:
                keep.append(v)
        self.pending = keep
        self.prev_tracks = tracks

    # ---------------------------------------------------------------- taste for the brain
    def taste_inject(self) -> list:
        if not self.cfg["taste"] or not self.cells:
            return []
        out = []
        gain = self.cfg["stimulus"].get("taste", 1.0)
        pos, neg = gain * min(1.0, self.taste_pos / 3), gain * min(1.0, self.taste_neg / 3)
        if pos > 0.02:
            out += [(self.cells["sweet"], np.float32(TASTE_MAX * pos)), (self.cells["pam"], np.float32(TASTE_MAX * pos))]
        if neg > 0.02:
            out += [(self.cells["bitter"], np.float32(TASTE_MAX * neg)), (self.cells["ppl1"], np.float32(TASTE_MAX * neg))]
        return [(c, a) for c, a in out if len(c)]

    # ---------------------------------------------------------------- is a detector misfiring in this game?
    def advice(self, per: dict) -> list[dict]:
        out = []
        w = self.cfg["weights"]
        play = sum(1 for _, p, _ in self.ticks_seen if p)
        if play < 400:                                   # need ~20 s of play in the last minute
            return out
        n = lambda ev: per.get(ev, {}).get("count", 0)
        if w.get("consume") and n("consume") > 20:
            out.append({"event": "consume", "weight": 0.0, "message":
                        f"Pickups fired {n('consume')} times in the last minute. In this game that is probably "
                        "pieces of sprites or effects vanishing near the player, not items."})
        on = sum(1 for _, p, d in self.ticks_seen if p and d) / play
        if w.get("danger") and on > 0.85:
            out.append({"event": "danger", "weight": 0.0, "message":
                        f"Danger was on {on:.0%} of the time: something always sits near the player (scenery, "
                        "the road, its own shots), so it only adds a constant punishment."})
        if w.get("food") and self.score_gains == 0 and self.play_ticks >= 3600 and self.p.get("score"):
            out.append({"event": "food", "weight": None, "message":
                        f"No score change in {self.play_ticks // 1200} minutes of play. If the score really went "
                        "up, its RAM address is wrong for this game."})
        return out

    # ---------------------------------------------------------------- for the dashboard
    def state(self) -> dict:
        per = {}
        for _, ev, r in self.recent:
            e = per.setdefault(ev, {"count": 0, "reward": 0.0})
            e["count"] += 1
            e["reward"] += r
        return {"config": {**self.cfg, "ram_rules": [{**r, "addr": f"{r['addr']:#05x}"} for r in self.cfg["ram_rules"]]},
                "advice": self.advice(per), "suggestions": self.scout.suggestions,
                "last_minute": {k: {"count": v["count"], "reward": round(v["reward"], 1)} for k, v in per.items()},
                "log": list(self.log), "taste": {"sweet": round(min(1.0, self.taste_pos / 3), 2),
                                                 "bitter": round(min(1.0, self.taste_neg / 3), 2)}}


class RamScout:
    """Watches game memory while the fly plays and suggests RAM rules: bytes that behave like a
    collectible counter (goes up in small steps, rarely changes, never counts down, and mostly goes up
    when the score does - collecting usually scores) or like health (drops in small steps shortly
    before most lost lives, needs 3+, and also on hits that don't kill). Score, lives, the stack and the sprite table are left out.
    Snapshots every 20 ticks (1 s), up to 15 minutes; re-analysed every minute."""

    EVERY, KEEP, ANALYSE = 20, 900, 60

    def __init__(self, profile: dict):
        skip = set(range(0x100, 0x300))                  # stack, shadow sprite table
        sc = profile.get("score")
        if sc:
            skip |= set(range(sc["addr"], sc["addr"] + sc["n"]))
        if profile.get("lives") is not None:
            skip.add(profile["lives"])
        if profile.get("progress"):
            skip |= {profile["progress"]["hi"], profile["progress"]["lo"]}
        self.keep_addr = np.array([a for a in range(0x800) if a not in skip])
        self.snaps: list[np.ndarray] = []
        self.pains: list[int] = []
        self.gains: list[int] = []                       # snapshots in which the score went up
        self.tick = 0
        self.pain_now = self.gain_now = False
        self.suggestions: list[dict] = []

    def observe(self, ram, playing: bool, pain: bool, gain: bool = False) -> None:
        self.tick += 1
        self.pain_now |= pain
        self.gain_now |= gain
        if self.tick % self.EVERY or not playing:
            return
        self.snaps.append(np.array(ram[:0x800], np.uint8)[self.keep_addr])
        if self.pain_now:
            self.pains.append(len(self.snaps) - 1)
        if self.gain_now:
            self.gains.append(len(self.snaps) - 1)
        self.pain_now = self.gain_now = False
        if len(self.snaps) > self.KEEP:
            self.snaps.pop(0)
            self.pains = [p - 1 for p in self.pains if p > 0]
            self.gains = [g - 1 for g in self.gains if g > 0]
        if len(self.snaps) >= self.ANALYSE and len(self.snaps) % self.ANALYSE == 0:
            self.suggestions = self.analyse()

    def analyse(self) -> list[dict]:
        V = np.array(self.snaps, np.int16)                       # (snapshots, addresses)
        D = np.diff(V, axis=0)
        changes = (D != 0).mean(axis=0)
        ups = ((D > 0) & (D <= 3)).sum(axis=0)
        bad_down = ((D < 0) & (V[1:] != 0)).sum(axis=0)          # falls that are not a reset to 0
        downs = ((D < 0) & (D >= -4)).sum(axis=0)
        out = []
        scored = np.zeros(len(V), bool)                          # snapshot t: the score rose at t-1, t or t+1
        for g in self.gains:
            scored[max(0, g - 1):g + 2] = True
        up_mask = (D > 0) & (D <= 3)                             # row t-1 of D: change from t-1 to t
        with_score = (up_mask & scored[1:, None]).sum(axis=0) / np.maximum(ups, 1)
        # score displays elsewhere (top score, a second copy): runs of 3+ neighbouring bytes holding 0-9
        digit = (V.max(axis=0) <= 9) & (changes > 0)
        addr = self.keep_addr
        run = np.zeros_like(digit)
        for k in range(len(digit)):
            if digit[k]:
                nb = [m for m in (k - 2, k - 1, k + 1, k + 2) if 0 <= m < len(digit) and abs(int(addr[m]) - int(addr[k])) <= 2]
                run[k] = sum(digit[m] for m in nb) >= 2
        counter = (ups >= 4) & (bad_down <= 1) & (changes < 0.15) & (with_score >= 0.75) & ~run
        for j in np.argsort(-ups * counter)[:4]:
            if counter[j]:
                out.append(self._row(j, V, "counter", f"went up {int(ups[j])} times in small steps, "
                                     f"{with_score[j]:.0%} of them with the score", {"when": "up", "weight": 0.5}))
        if len(self.pains) >= 3:
            hits = np.zeros(V.shape[1], int)
            for p in self.pains:
                lo = max(0, p - 6)
                if p > lo:
                    hits += ((D[lo:p] < 0) & (D[lo:p] >= -4)).any(axis=0)
            # real health also drops on hits that don't kill: a byte that drops only at deaths is a
            # death marker (it resets when the player dies), not health
            # ...and like health in any game: back to its maximum soon after most lost lives (refilled
            # for the new life), at least 2 points, losing them mostly one at a time
            top = V.max(axis=0)
            refill = np.zeros(V.shape[1], int)
            for p in self.pains:
                refill += (V[min(len(V) - 1, p + 1):p + 6] == top).any(axis=0) if p + 1 < len(V) else 0
            ones = (D == -1).sum(axis=0) / np.maximum(downs, 1)
            health = ((hits >= 0.6 * len(self.pains)) & (downs >= hits + 2) & (changes < 0.35) & ~counter & ~run
                      & (refill >= 0.6 * len(self.pains)) & (top >= 2) & (ones >= 0.7))
            for j in np.argsort(-hits * health)[:3]:
                if health[j]:
                    out.append(self._row(j, V, "health", f"dropped before {int(hits[j])} of {len(self.pains)} lost lives",
                                         {"when": "down", "weight": -2.0}))
        return out

    def _row(self, j, V, kind, why, suggest) -> dict:
        vals = list(dict.fromkeys(int(v) for v in V[:, j]))[-6:]
        return {"addr": f"{int(self.keep_addr[j]):#05x}", "kind": kind, "why": why, "values": vals, **suggest}
