"""100 famous NES games and how the fly should hold the controller for each.

Matched by ROM file name ("Super Mario Bros. (World).nes" -> super mario bros).
A genre sets the controls; a game can override single fields.

    axes     which d-pad axes the fly steers: "H" left/right, "V" up/down
    fire     button for shooting / attacking (the learner's fire head, the strike rule)
    jump     button pressed when the escape readouts fire (the giant fibre DNp01 is
             the fly's own take-off jump), or None
    hold     buttons held down the whole time (e.g. accelerate)
    menu     keys tried, in turn, on title screens and menus (default START, START, B, A;
             Double Dragon needs SELECT, which in Excitebike would pick the track editor)
    menu_every  seconds between those presses (racers: long, so a start countdown is not paused)
    ok       False where a fly's reflexes can't play the game (menus, puzzles, light gun)
    forward  the way the level goes (platformers: right); the fly's idle wandering heads there

No ROMs are included; use your own copies.
"""
from __future__ import annotations

import re

MENU = ("start", "start", "b", "a")

GENRES = {
    "shooter_v": {"axes": "H", "fire": "a", "jump": None, "hold": (), "ok": True},        # ship at the bottom
    "shooter_8": {"axes": "HV", "fire": "a", "jump": None, "hold": (), "ok": True},       # free-flying shooters
    "shooter_h": {"axes": "HV", "fire": "b", "jump": None, "hold": (), "ok": True},       # side-scrolling shooters
    "platformer": {"axes": "H", "fire": "b", "jump": "a", "hold": (), "ok": True, "forward": "right"},
    "run_gun": {"axes": "H", "fire": "b", "jump": "a", "hold": (), "ok": True, "forward": "right"},
    "topdown": {"axes": "HV", "fire": "b", "jump": None, "hold": (), "ok": True},
    "maze": {"axes": "HV", "fire": None, "jump": None, "hold": (), "ok": True},
    "beat_em_up": {"axes": "HV", "fire": "b", "jump": None, "hold": (), "ok": True},
    "racing": {"axes": "H", "fire": None, "jump": None, "hold": ("a",), "ok": True, "menu_every": 12.0},
    "fighting": {"axes": "H", "fire": "b", "jump": None, "hold": (), "ok": True},
    "sports": {"axes": "HV", "fire": "a", "jump": None, "hold": (), "ok": False},
    "rpg": {"axes": "HV", "fire": "a", "jump": None, "hold": (), "ok": False},
    "puzzle": {"axes": "H", "fire": "a", "jump": None, "hold": (), "ok": False},
    "light_gun": {"axes": "", "fire": None, "jump": None, "hold": (), "ok": False},
}

# name: (genre, overrides)
GAMES = {
    "super mario bros": ("platformer", {}),
    "super mario bros 2": ("platformer", {}),
    "super mario bros 3": ("platformer", {}),
    "the legend of zelda": ("topdown", {"fire": "a"}),
    "zelda ii the adventure of link": ("platformer", {"fire": "b"}),
    "metroid": ("run_gun", {}),
    "mega man": ("run_gun", {}),
    "mega man 2": ("run_gun", {}),
    "mega man 3": ("run_gun", {}),
    "mega man 4": ("run_gun", {}),
    "mega man 5": ("run_gun", {}),
    "mega man 6": ("run_gun", {}),
    "castlevania": ("platformer", {}),
    "castlevania ii simon s quest": ("platformer", {}),
    "castlevania iii dracula s curse": ("platformer", {}),
    "contra": ("run_gun", {}),
    "super c": ("run_gun", {}),
    "ninja gaiden": ("platformer", {}),
    "ninja gaiden ii the dark sword of chaos": ("platformer", {}),
    "ninja gaiden iii the ancient ship of doom": ("platformer", {}),
    "kid icarus": ("run_gun", {}),
    "duck tales": ("platformer", {}),
    "chip n dale rescue rangers": ("platformer", {}),
    "kirby s adventure": ("platformer", {}),
    "bionic commando": ("run_gun", {}),
    "blaster master": ("run_gun", {}),
    "ghosts n goblins": ("run_gun", {}),
    "adventure island": ("platformer", {}),
    "little nemo the dream master": ("platformer", {}),
    "batman": ("platformer", {}),
    "teenage mutant ninja turtles": ("platformer", {}),
    "teenage mutant ninja turtles ii the arcade game": ("beat_em_up", {}),
    "double dragon": ("beat_em_up", {"menu": ("start", "start", "select", "start")}),
    "double dragon ii the revenge": ("beat_em_up", {}),
    "river city ransom": ("beat_em_up", {}),
    "battletoads": ("beat_em_up", {}),
    "kung fu": ("fighting", {"axes": "H"}),
    "mike tyson s punch out": ("fighting", {"axes": ""}),
    "punch out": ("fighting", {"axes": ""}),
    "metal gear": ("topdown", {}),
    "ikari warriors": ("topdown", {"fire": "b"}),
    "commando": ("topdown", {}),
    "jackal": ("topdown", {}),
    "life force": ("shooter_h", {}),
    "gradius": ("shooter_h", {}),
    "1942": ("shooter_8", {"fire": "b"}),
    "1943 the battle of midway": ("shooter_8", {"fire": "b"}),
    "xevious": ("shooter_8", {}),
    "galaga": ("shooter_v", {}),
    "galaxian": ("shooter_v", {}),
    "space invaders": ("shooter_v", {}),
    "twinbee": ("shooter_8", {}),
    "star force": ("shooter_8", {}),
    "gun smoke": ("shooter_8", {}),
    "sky kid": ("shooter_h", {}),
    "pac man": ("maze", {}),
    "ms pac man": ("maze", {}),
    "dig dug": ("maze", {"fire": "a"}),
    "bomberman": ("maze", {"fire": "a"}),
    "lode runner": ("maze", {"fire": "a"}),
    "donkey kong": ("platformer", {"axes": "HV", "fire": None}),
    "donkey kong jr": ("platformer", {"axes": "HV", "fire": None}),
    "mario bros": ("platformer", {"fire": None}),
    "balloon fight": ("platformer", {"jump": "a", "fire": None}),
    "ice climber": ("platformer", {}),
    "joust": ("platformer", {"fire": None}),
    "bubble bobble": ("platformer", {}),
    "rygar": ("platformer", {}),
    "dragon warrior": ("rpg", {}),
    "final fantasy": ("rpg", {}),
    "crystalis": ("topdown", {}),
    "startropics": ("topdown", {}),
    "excitebike": ("racing", {"axes": "HV"}),
    "r c pro am": ("racing", {"axes": "H"}),
    "rad racer": ("racing", {}),
    "micro machines": ("racing", {}),
    "mach rider": ("racing", {}),
    "paperboy": ("racing", {"axes": "HV", "fire": "b"}),
    "tetris": ("puzzle", {}),
    "dr mario": ("puzzle", {}),
    "yoshi": ("puzzle", {}),
    "duck hunt": ("light_gun", {}),
    "hogan s alley": ("light_gun", {}),
    "tecmo bowl": ("sports", {}),
    "tecmo super bowl": ("sports", {}),
    "ice hockey": ("sports", {}),
    "tennis": ("sports", {}),
    "golf": ("sports", {}),
    "baseball": ("sports", {}),
    "rbi baseball": ("sports", {}),
    "track field": ("sports", {}),
    "pinball": ("sports", {}),
    "arkanoid": ("puzzle", {"ok": True, "fire": "a"}),
    "gauntlet": ("topdown", {"fire": "a"}),
    "ghostbusters": ("topdown", {}),
    "top gun": ("shooter_8", {}),
    "section z": ("shooter_h", {}),
    "solomon s key": ("platformer", {}),
    "wizards warriors": ("platformer", {}),
    "shadow of the ninja": ("platformer", {}),
}

TAGS = re.compile(r"\((?:[^)]*)\)|\[(?:[^]]*)\]")


def normalize(name: str) -> str:
    name = TAGS.sub(" ", name.lower()).replace("&", " ")
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return " ".join(w for w in name.split() if w not in ("the", "a", "an"))


KEYS = {normalize(g): g for g in GAMES}


def controls_for(rom_name: str) -> dict:
    """Controls for this ROM (longest catalogue name contained in the file name); unknown
    games get top-down controls, which suit the widest range."""
    n = f" {normalize(rom_name)} "
    best = max((k for k in KEYS if f" {k} " in n), key=len, default=None)
    genre, over = GAMES[KEYS[best]] if best else ("topdown", {})
    c = {"menu": MENU, "menu_every": 2.0, **GENRES[genre], **over}
    c.update(game=KEYS[best] if best else "unknown", genre=genre)
    return c
