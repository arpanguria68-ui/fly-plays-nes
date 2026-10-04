"""Tests for the NES side (mesen/): controls, tracking, menus, rewards, eyes.

    python -m pytest tests -q

No GPU and no brain are needed, except the eye-map test, which runs when the brain
files are present (it reads the connectome)."""
from __future__ import annotations

import sys
import zipfile
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mesen"))

import fly_nes  # noqa: E402
from fly_dashboard import nes_inside  # noqa: E402
from games import controls_for  # noqa: E402
from learner import Reward, profile_for, read_score  # noqa: E402

SMB = profile_for("Super Mario Bros. (Europe)")


# ---------------------------------------------------------------- games and rewards

def test_profiles_match_the_right_games():
    assert profile_for("Super Mario Bros. (Europe)") is SMB
    assert profile_for("Super Mario Bros. _ Duck Hunt") is SMB
    assert profile_for("Super Mario Bros. 3 (USA)") is None        # other RAM layout
    assert profile_for("super mario bros 2") is None
    assert profile_for("Galaga (U)")["lives"] == 0x485


def test_read_score_digits_and_bcd():
    ram = np.zeros(0x800, np.uint8)
    ram[0x7DD:0x7E3] = [0, 0, 1, 7, 0, 0]
    assert read_score(ram, SMB["score"]) == 1700
    ram[0x10:0x13] = [0x01, 0x23, 0x45]
    assert read_score(ram, {"addr": 0x10, "n": 3, "fmt": "bcd", "order": "big"}) == 12345
    ram[0x7DD] = 12                                                 # not a digit: not a score
    assert read_score(ram, SMB["score"]) is None


def smb_ram(page=0, x=0, lives=2, score=(0, 0, 0, 0, 0, 0)):
    ram = np.zeros(0x800, np.uint8)
    ram[0x6D], ram[0x86], ram[0x75A] = page, x, lives
    ram[0x7DD:0x7E3] = score
    return ram


def test_progress_rewards_new_ground_only():
    r = Reward(SMB)
    r(smb_ram(x=10))
    assert r(smb_ram(x=90)) == pytest.approx(80 / 160)
    assert r(smb_ram(x=40)) == 0                                    # walking back
    assert r(smb_ram(x=90)) == 0                                    # ...and over old ground again
    assert r(smb_ram(x=110)) == pytest.approx(20 / 160)


def test_death_costs_and_restart_is_not_progress():
    r = Reward(SMB)
    r(smb_ram(page=3, x=0, lives=2))
    assert r(smb_ram(page=0, x=20, lives=1)) == pytest.approx(-10)  # back to the start, a life lost
    assert r(smb_ram(page=0, x=40, lives=1)) == pytest.approx(20 / 160)


def test_score_reward_in_units():
    r = Reward(SMB)
    r(smb_ram(score=(0, 0, 0, 0, 0, 0)))
    assert r(smb_ram(score=(0, 0, 0, 0, 1, 0))) == pytest.approx(1.0)   # 100 points on screen


def test_controls_for_catalogue_and_default():
    c = controls_for("Super Mario Bros. 3 (USA)")
    assert c["genre"] == "platformer" and c["forward"] == "right" and c["jump"] == "a"
    assert controls_for("Double Dragon (USA)")["menu"] == ("start", "start", "select", "start")
    unknown = controls_for("Some Homebrew Game")
    assert unknown["axes"] and unknown["menu_every"] > 0


def test_checkpoint_name_per_motion_mode():
    assert fly_nes.checkpoint_name("Galaga (U)", "blobs") == "Galaga (U).npz"
    assert fly_nes.checkpoint_name("Galaga (U)", "columns") == "Galaga (U).columns.npz"
    assert fly_nes.checkpoint_name("Galaga (U)", "blobs", "td") == "Galaga (U).td.npz"


def test_library_skips_zips_without_a_nes_rom(tmp_path):
    good, gba = tmp_path / "a.zip", tmp_path / "b.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("game.nes", b"NES\x1a")
    with zipfile.ZipFile(gba, "w") as z:
        z.writestr("game.gba", b"x")
    assert nes_inside(good) and not nes_inside(gba)
    assert nes_inside(tmp_path / "plain.nes")


# ---------------------------------------------------------------- sprites and tracking

def test_cluster_groups_a_metasprite():
    tiles = [{"x": 100 + dx, "y": 100 + dy, "tile": 1 + k, "attr": 0}
             for k, (dx, dy) in enumerate([(0, 0), (8, 0), (0, 8), (8, 8)])]
    objs = fly_nes.cluster_sprites(tiles)
    assert len(objs) == 1 and objs[0]["w"] == 16 and objs[0]["h"] == 16


def test_tracker_finds_the_sprite_that_follows_the_pad():
    t = fly_nes.Tracker()
    x = 50.0
    player = None
    for k in range(20):
        x += 2
        objs = [{"x": x, "y": 100.0, "w": 16.0, "h": 16.0, "n": 4},               # moves with "right"
                {"x": 200.0 - 2 * (k % 3), "y": 60.0, "w": 16.0, "h": 16.0, "n": 4}]  # drifts on its own
        player, _ = t.update(objs, (1, 0))
    assert player["y"] == 100.0


def test_fixed_objects_are_not_targets():
    t = fly_nes.Tracker()
    x = 50.0
    others = []
    for _ in range(fly_nes.FIXED_TICKS + 5):
        x = 60.0 if x == 50.0 else 50.0
        objs = [{"x": x, "y": 150.0, "w": 16.0, "h": 16.0, "n": 4},
                {"x": 92.0, "y": 28.0, "w": 8.0, "h": 8.0, "n": 1}]                   # a status-bar icon
        _, others = t.update(objs, (1 if x == 60.0 else -1, 0))
    assert not any(o["y"] == 28.0 for o in others)


def test_drawn_accepts_small_mario_with_blank_tiles():
    import server_emu as se
    frame = np.full((240, 256, 3), 100, np.uint8)
    spr = [{"i": i, "x": 20 + 8 * (i % 3), "y": 100 + 8 * (i // 3), "tile": 0, "attr": 0} for i in range(9)]
    for s in spr[:5]:                                              # 5 of 9 sprites visible
        frame[s["y"] + 2:s["y"] + 6, s["x"]:s["x"] + 6] = 250
    assert se.drawn(spr, frame)
    frame[:] = 100
    for s in spr[:2]:
        frame[s["y"] + 2:s["y"] + 6, s["x"]:s["x"] + 6] = 250
    assert not se.drawn(spr, frame)


# ---------------------------------------------------------------- menus: the stuck deadlock

def menu_fly(game: str = "Double Dragon (USA)"):
    """Just the state _menus() uses, without a brain."""
    f = fly_nes.FlyNes.__new__(fly_nes.FlyNes)
    f.controls = controls_for(game)
    f.tracker = SimpleNamespace(player_agree=None, responded=False)
    f.last_pad = dict.fromkeys(fly_nes.PAD_KEYS, False)
    f.press_key, f.press_run, f.reversed = None, 0, False
    f.agreement, f.tried = {}, set()
    f.scrolling = deque(maxlen=40)
    f.race_block_until, f.race_ok, f.race_check = 0.0, False, None
    f.in_control, f.unresponsive, f.still_for, f.seeking = False, 0.0, 0.0, True
    f.unstick_left, f.unstick_i, f.next_unstick, f.menu_waits, f.probe_tick = 0, 0, 0.0, 0, 0
    f.explore, f.since_new = None, 0.0
    f.recent_follow = deque(maxlen=fly_nes.FOLLOW_WINDOW)
    f.wiggle, f.next_wiggle, f.lost_control = None, 0.0, False
    f.last_player = {"x": 100.0, "y": 200.0}
    return f


def test_stuck_does_not_lock_out_control():
    f = menu_fly()
    f.unresponsive, f.tried = 10.0, {"up", "down", "left", "right"}
    f._menus(dict.fromkeys(fly_nes.PAD_KEYS, False), 1.0, 0.05, True)
    assert f.unresponsive == 0.0 and not f.tried                   # stuck restarted its counters
    now = 2.0
    for k in range(60):                                            # the player now follows the pad
        key = ("left", "right", "up", "down")[(k // 10) % 4]
        f.last_pad = dict.fromkeys(fly_nes.PAD_KEYS, False)
        f.last_pad[key] = True
        f.tracker.player_agree, f.tracker.responded = 1.0, True
        now += 0.05
        f._menus(dict.fromkeys(fly_nes.PAD_KEYS, False), now, 0.05, True)
    assert f.in_control and not f.seeking


def test_left_right_game_notices_losing_control():
    f = menu_fly("Galaga (U)")                                     # steers left/right only
    f.in_control, f.seeking = True, False
    f.agreement = {"left": deque([True] * 3, maxlen=10), "right": deque([True] * 3, maxlen=10)}
    now = 1.0
    for k in range(240):                                           # an attract demo: the ship ignores us (12 s)
        f.last_pad = dict.fromkeys(fly_nes.PAD_KEYS, False)
        f.last_pad[("left", "right")[(k // 10) % 2]] = True
        f.tracker.player_agree, f.tracker.responded = None, False
        now += 0.05
        f._menus(dict.fromkeys(fly_nes.PAD_KEYS, False), now, 0.05, True)
    assert f.seeking and not f.in_control


def run_in_control(f, match_rate, ours, ticks=600, seed=0):
    """Hold left/right in turn; the tracked sprite matches the press at match_rate. If `ours`,
    it moves the way we press (a wiggle check passes); otherwise it drifts on its own.
    Returns whether control was kept."""
    rng = np.random.default_rng(seed)
    f.in_control, f.seeking = True, False
    now, x = 1.0, 100.0
    for k in range(ticks):
        pressed = [d for d in ("left", "right") if f.last_pad[d]]
        if pressed and (ours if f.wiggle is not None else rng.random() < match_rate):
            x += 4.0 if pressed[0] == "right" else -4.0
        else:
            x += 4.0 * (1 if rng.random() < 0.5 else -1) if not ours else 0.0
        f.last_player = {"x": x, "y": 200.0}
        if pressed and f.wiggle is None:
            match = rng.random() < match_rate
            f.tracker.player_agree, f.tracker.responded = (1.0 if match else -1.0), match
        else:
            f.tracker.player_agree, f.tracker.responded = None, False
        now += 0.05
        out = f._menus(dict.fromkeys(fly_nes.PAD_KEYS, False), now, 0.05, bool(pressed))
        if f.seeking:
            return False
        if f.wiggle is not None or any(out.values()):
            f.last_pad = out                                           # the check drives the pad
        else:
            f.last_pad = dict.fromkeys(fly_nes.PAD_KEYS, False)
            f.last_pad[("left", "right")[(k // 10) % 2]] = True
    return True


def test_demo_that_sometimes_matches_the_pad_is_noticed():
    assert not run_in_control(menu_fly("Galaga (U)"), match_rate=0.55, ours=False)


def test_real_play_keeps_control():
    assert run_in_control(menu_fly("Galaga (U)"), match_rate=0.95, ours=True)
    # doubtful follow rate (skids), but it goes both ways when wiggled: still ours
    assert run_in_control(menu_fly("Galaga (U)"), match_rate=0.7, ours=True)


def test_menu_key_held_back_while_probes_move_the_player():
    f = menu_fly()
    f.next_unstick = 1.0
    f.agreement = {"left": deque([True, True], maxlen=10)}
    f.tracker.player_agree = None
    pad = f._menus(dict.fromkeys(fly_nes.PAD_KEYS, False), 1.2, 0.05, False)
    assert not pad["start"] and f.menu_waits == 1                   # no START: it would pause the game


# ---------------------------------------------------------------- eyes

def test_motion_detector_direction():
    img0 = np.zeros((20, 40), np.float32)
    img1 = np.zeros((20, 40), np.float32)
    img0[:, 10] = 1.0                                               # a bright bar...
    img1[:, 12] = 1.0                                               # ...one tick later, 2 px right
    h, v = fly_nes.ColumnEyes._correlate(img0, img1)
    assert h[:, 9:14].sum() > 0 and abs(v).sum() < 1e-6
    h_back, _ = fly_nes.ColumnEyes._correlate(img1, img0)           # the same, moving left
    assert h_back[:, 9:14].sum() < 0


def test_eye_map_directions_from_the_wiring():
    from flybrain.data import DATA, has_data
    if not has_data(DATA):
        pytest.skip("brain files not present")
    from flybrain.columns import eye_map
    m = eye_map()
    for t, want in (("T4a", (-1, 0)), ("T4b", (1, 0)), ("T4c", (0, 1)), ("T4d", (0, -1))):
        for side in "LR":
            p = m["pref"][(m["types"] == t) & (m["eye"] == side)].mean(axis=0)
            assert np.dot(p, want) > 0.8, (t, side, p)


# ---------------------------------------------------------------- reward feed

def feed_fly(tracks=(), player=None, seeking=False):
    return SimpleNamespace(seeking=seeking, last_player=player,
                           tracker=SimpleNamespace(tracks=list(tracks), player_id=0))


def make_feed(tmp_path, monkeypatch, genre="platformer"):
    import reward_feed as rf
    monkeypatch.setattr(rf, "RULES_DIR", tmp_path)
    return rf, rf.RewardFeed(SMB, {"genre": genre}, "Test Game")


def test_feed_food_pain_explore(tmp_path, monkeypatch):
    rf, f = make_feed(tmp_path, monkeypatch)
    fly = feed_fly()
    f(smb_ram(x=10, score=(0, 0, 0, 0, 0, 0)), fly)
    assert f(smb_ram(x=10, score=(0, 0, 0, 0, 1, 0)), fly) == pytest.approx(1.0)       # food: +100 on screen
    assert f(smb_ram(x=170, score=(0, 0, 0, 0, 1, 0)), fly) == pytest.approx(1.0)      # explore: 160 px
    assert f(smb_ram(x=170, lives=1, score=(0, 0, 0, 0, 1, 0)), fly) == pytest.approx(-10.0)   # pain
    assert [e["event"] for e in f.log][:3] == ["pain", "explore", "food"]


def test_feed_nothing_counts_in_menus(tmp_path, monkeypatch):
    rf, f = make_feed(tmp_path, monkeypatch)
    f(smb_ram(score=(0, 0, 0, 0, 0, 0)), feed_fly(seeking=True))
    assert f(smb_ram(score=(0, 0, 0, 0, 5, 0)), feed_fly(seeking=True)) == 0      # an attract demo scoring


def test_feed_kill_and_relief(tmp_path, monkeypatch):
    rf, f = make_feed(tmp_path, monkeypatch, genre="shooter_v")
    me = {"id": 0, "x": 100.0, "y": 200.0}
    enemy = {"id": 7, "x": 100.0, "y": 165.0, "still": 0, "age": 40}
    f(smb_ram(score=(0, 0, 0, 0, 0, 0)), feed_fly([me, enemy], me))
    r = f(smb_ram(score=(0, 0, 0, 0, 1, 0)), feed_fly([me], me))        # the enemy is gone, the score rose
    w = f.cfg["weights"]
    assert r == pytest.approx(w["food"] + w["kill"] + w["relief"] * (1 - 35 / rf.THREAT_PX))


def test_feed_consume_needs_survival(tmp_path, monkeypatch):
    rf, f = make_feed(tmp_path, monkeypatch)
    me = {"id": 0, "x": 100.0, "y": 200.0}
    item = {"id": 3, "x": 105.0, "y": 200.0, "still": 0, "age": 40}
    f(smb_ram(), feed_fly([me, item], me))
    rewards = [f(smb_ram(), feed_fly([me], me)) for _ in range(rf.CONFIRM_TICKS + 1)]
    assert any(e["event"] == "consume" for e in f.log)
    rf2, g = make_feed(tmp_path, monkeypatch)
    g(smb_ram(lives=2), feed_fly([me, item], me))
    g(smb_ram(lives=1), feed_fly([me], me))                              # it killed us: not food
    for _ in range(rf.CONFIRM_TICKS + 1):
        g(smb_ram(lives=1), feed_fly([me], me))
    assert not any(e["event"] == "consume" for e in g.log)


def test_feed_ram_rule_and_saved_config(tmp_path, monkeypatch):
    rf, f = make_feed(tmp_path, monkeypatch)
    cfg = rf.load_config("Test Game", "platformer")
    cfg["ram_rules"] = rf.clean_rules([{"name": "coins", "addr": "0x75E", "when": "up", "weight": 0.5},
                                       {"name": "bad", "addr": "0x900", "when": "up"}])
    rf.save_config("Test Game", cfg)
    f.set_config(rf.load_config("Test Game", "platformer"))
    assert [r["name"] for r in f.cfg["ram_rules"]] == ["coins"]
    ram = smb_ram()
    f(ram, feed_fly())
    ram2 = ram.copy(); ram2[0x75E] = 1
    assert f(ram2, feed_fly()) == pytest.approx(0.5)


def test_feed_ignores_short_lived_pieces(tmp_path, monkeypatch):
    rf, f = make_feed(tmp_path, monkeypatch)
    me = {"id": 0, "x": 100.0, "y": 200.0}
    piece = {"id": 9, "x": 104.0, "y": 190.0, "still": 0, "age": 3}     # part of the player's sprite, briefly split off
    f(smb_ram(), feed_fly([me, piece], me))
    for _ in range(rf.CONFIRM_TICKS + 2):
        f(smb_ram(), feed_fly([me], me))
    assert not f.log


def test_feed_one_kill_per_score_gain(tmp_path, monkeypatch):
    rf, f = make_feed(tmp_path, monkeypatch, genre="shooter_v")
    me = {"id": 0, "x": 100.0, "y": 200.0}
    foes = [{"id": k, "x": 60.0 + 20 * k, "y": 100.0, "still": 0, "age": 40} for k in (1, 2, 3)]
    f(smb_ram(score=(0, 0, 0, 0, 0, 0)), feed_fly([me] + foes, me))
    f(smb_ram(score=(0, 0, 0, 0, 1, 0)), feed_fly([me], me))           # three vanish, one score gain
    for _ in range(rf.MATCH_TICKS + 1):
        f(smb_ram(score=(0, 0, 0, 0, 1, 0)), feed_fly([me], me))
    assert sum(e["event"] == "kill" for e in f.log) == 1


# ---------------------------------------------------------------- learner v2 (n-step TD)

def test_old_checkpoint_is_kept_not_overwritten(tmp_path):
    from learner import Learner
    heads = {"H": ("left", None, "right")}
    v1 = Learner(4, tmp_path / "g.npz", "g", heads, td=False)
    v1.XtX["H"] += np.eye(4); v1.samples = 123
    v1.save()
    v2 = Learner(4, tmp_path / "g.npz", "g", heads, td=True)
    assert v2.load() is False                                 # different learner: start fresh...
    assert (tmp_path / "g.v1.npz").is_file()                  # ...but the old learning is kept
    assert int(np.load(tmp_path / "g.v1.npz")["samples"]) == 123


def test_td_credit_reaches_past_one_second(tmp_path):
    """A reward 2 s after a choice: version 1 (1 s window) never links them; TD does, via its
    own estimate of what follows."""
    import learner as L
    heads = {"H": ("left", None, "right")}
    def run(td):
        lr = L.Learner(2, tmp_path / f"t{td}.npz", "t", heads, td=td)
        lr.epsilon = lambda: 0.0
        x_cue, x_wait = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        for episode in range(400):
            for k in range(60):                               # cue at k=0, reward 40 ticks (2 s) later
                x = x_cue if k == 0 else x_wait
                lr.step({"H": x}, {"left": False, "right": False}, 1.0 if k == 40 else 0.0)
        for h in lr.heads:
            lr.refit(h)
        return float((lr.w["H"] @ x_cue).max())
    assert run(False) < 0.05 < run(True)


# ---------------------------------------------------------------- adaptive: RAM scout, advice, stimulus gains

def test_scout_finds_a_coin_counter_and_health():
    import reward_feed as rf
    scout = rf.RamScout(SMB)
    rng = np.random.default_rng(0)
    ram = np.zeros(0x800, np.uint8)
    coins, health, t = 0, 5, 0
    for second in range(130):
        coin = second % 9 == 4
        if coin:
            coins += 1                                     # a coin now and then (it scores)
        if second % 20 in (5, 10, 15):
            health -= 1                                    # hits (the last one fatal)...
        pain = second % 20 == 17 and health < 5
        if pain:
            health = 5                                     # ...then a life lost, health refilled
        ram[0x75E], ram[0x600] = coins, health
        ram[0x7D9:0x7DC] = [0, (second // 9) % 10, 0]      # a score display: digits that rise with the score
        ram[0x10] = rng.integers(256)                      # noise byte (a timer or random number)
        ram[0x20] = second // 15                           # a level counter: rises, but never with the score
        for k in range(rf.RamScout.EVERY):
            scout.observe(ram, True, pain and k == 0, coin and k == 0)
    found = {(s["addr"], s["kind"]) for s in scout.suggestions}
    assert ("0x75e", "counter") in found
    assert ("0x600", "health") in found
    assert not any(s["addr"] in ("0x010", "0x020", "0x7da") for s in scout.suggestions)


def test_advice_flags_danger_that_never_switches_off(tmp_path, monkeypatch):
    rf, f = make_feed(tmp_path, monkeypatch, genre="shooter_v")
    me = {"id": 0, "x": 100.0, "y": 200.0}
    scenery = {"id": 5, "x": 110.0, "y": 200.0, "still": 0, "age": 100}
    for _ in range(500):
        f(smb_ram(), feed_fly([me, scenery], me))
    assert any(a["event"] == "danger" for a in f.state()["advice"])


def test_stimulus_gain_scales_one_sense():
    f = fly_nes.FlyNes.__new__(fly_nes.FlyNes)
    loom = np.arange(5)
    chase = np.arange(5, 9)
    f.eyes = SimpleNamespace(cells={"loom": {"L": loom}, "chase": {"L": chase}})
    f.motion = SimpleNamespace(cells=np.arange(20, 30))
    f.stim_gain = {"loom": 2.0, "chase": 1.0, "eyes": 0.5}
    out = f._scale_senses([(loom, np.float32(0.3)), (chase, np.float32(0.3)), (f.motion.cells, np.float32(0.4))])
    assert [float(a) for _, a in out] == pytest.approx([0.6, 0.3, 0.2])


def test_stimulus_saved_and_clamped(tmp_path, monkeypatch):
    import reward_feed as rf
    monkeypatch.setattr(rf, "RULES_DIR", tmp_path)
    cfg = rf.load_config("G", "platformer")
    cfg["stimulus"].update(rf.clean_stimulus({"loom": 9, "taste": -1, "bogus": 2}))
    rf.save_config("G", cfg)
    back = rf.load_config("G", "platformer")["stimulus"]
    assert back["loom"] == rf.STIM_MAX and back["taste"] == 0.0 and "bogus" not in back


def test_scout_does_not_call_a_death_marker_health():
    import reward_feed as rf
    scout = rf.RamScout(SMB)
    ram = np.zeros(0x800, np.uint8)
    marker = 3
    for second in range(130):
        pain = second % 20 == 17
        if second % 20 == 16:
            marker = 0                                     # resets as the player dies...
        elif second % 20 == 3:
            marker = 3                                     # ...and is set again in the next life
        ram[0x650] = marker
        for k in range(rf.RamScout.EVERY):
            scout.observe(ram, True, pain and k == 0)
    assert not any(s["addr"] == "0x650" for s in scout.suggestions)


# ---------------------------------------------------------------- the dashboard pad (a person's buttons)
def test_human_pad_merges_and_wins_its_axis():
    fly = {k: False for k in fly_nes.PAD_KEYS} | {"right": True, "a": True}
    out = fly_nes.merge_human(fly, {"left": True}, drive=False)
    assert out["left"] and not out["right"] and out["a"]      # held left beats the fly's right, A stays
    out = fly_nes.merge_human(fly, {"b": True}, drive=True)
    assert out["b"] and not out["right"] and not out["a"]     # driving: only the person's buttons


def test_learner_credits_what_was_really_pressed(tmp_path):
    from learner import Learner
    heads = {"H": ("left", None, "right")}
    lr = Learner(2, tmp_path / "p.npz", "p", heads, td=False)
    lr.epsilon = lambda: 0.0
    lr.step({"H": np.ones(2)}, {"right": True}, 0.0)
    lr.relabel({"left": True, "right": False})
    assert lr.pending[-1][4] == 0                             # recorded as left, as the person pressed


# ---------------------------------------------------------------- adding your own ROMs from the Play page
def test_check_rom_accepts_only_nes_games():
    import io as _io
    from fly_dashboard import check_rom
    nes = b"NES\x1a" + bytes(16400)
    assert check_rom("game.nes", nes) is None
    assert check_rom("game.nes", b"PK\x03\x04junk") is not None              # no iNES header
    assert check_rom("game.gba", nes) is not None
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Game (USA).nes", nes)
    assert check_rom("game.zip", buf.getvalue()) is None
    assert check_rom("game.zip", b"not a zip") is not None


def test_add_rom_never_overwrites_another_game(tmp_path):
    from fly_dashboard import Dashboard
    d = SimpleNamespace(rom_dir=tmp_path, rom=tmp_path / "none.nes")
    d.library = lambda: {p.name: p for p in tmp_path.iterdir()}
    a, b = b"NES\x1a" + bytes(100), b"NES\x1a" + bytes(200)
    assert Dashboard.add_rom(d, "x.nes", a)["name"] == "x.nes"
    assert Dashboard.add_rom(d, "x.nes", a)["name"] == "x.nes"               # same game again: no copy
    assert Dashboard.add_rom(d, "../x.nes", b)["name"] == "x (2).nes"        # different game, same name
    assert not Dashboard.add_rom(d, "x.txt", a)["ok"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x (2).nes", "x.nes"]
