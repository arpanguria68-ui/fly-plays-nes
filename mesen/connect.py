"""Start the fly brain and its dashboard, with the game running in this process.

    pip install cynes                          # the NES emulator the server runs games with
    .\.venv\Scripts\python.exe mesen\connect.py --rom "roms\Your Game.nes"

Open http://127.0.0.1:8777/ to watch; the game keeps running if the tab is hidden or
closed. Pick another game from the list in the dashboard (the ROMs in the --rom
folder) and the fly adapts: its controls, its own learning checkpoint, menus.

GPU: with CuPy installed (`pip install -r requirements-gpu.txt`) the brain runs on
CUDA with 4 voting flies per axis; without it, on the CPU with 1 per axis (slower).
--engine browser runs the game in the page instead (EmulatorJS; pauses in background tabs).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.argv = [str(HERE / "fly_nes.py"), *sys.argv[1:]]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from fly_nes import main

if __name__ == "__main__":
    main()
