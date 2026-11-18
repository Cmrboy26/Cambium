import os

from arena.arenaClass import Arena as ac


CLASSIC_ARENAS = [
    ac([[-19, 14, 25], [-36, 14, 25]], "open", "Classic Arena 1",
       "classic", [-42, 14, 18, -13, 20, 31]),
    ac([[-19, 14, 37], [-36, 14, 37]], "open", "Classic Arena 2",
       "classic", [-42, 14, 32, -13, 20, 44]),
]

BRIDGE_ARENAS = [
    ac([[-19, 15, 25], [-36, 15, 25]], "open", "Bridge Duel 1",
       "bridge_duel",
       [-42, 14, 18, -13, 22, 32],
       [[-20, 14, 24, -18, 22, 26], [-37, 14, 24, -35, 22, 26]],
       [-19, 15, 25]),
    ac([[-19, 15, 37], [-36, 15, 37]], "open", "Bridge Duel 2",
       "bridge_duel",
       [-42, 14, 32, -13, 22, 44],
       [[-20, 14, 36, -18, 22, 38], [-37, 14, 36, -35, 22, 38]],
       [-19, 15, 37]),
]

MAP_PRESETS = {"classic": CLASSIC_ARENAS, "bridge_duel": BRIDGE_ARENAS}
DEFAULT_MAP = os.getenv("CAMBIUM_MAP", "bridge_duel").strip().lower()
if DEFAULT_MAP not in MAP_PRESETS:
    print(f"[ARENA] Unknown CAMBIUM_MAP '{DEFAULT_MAP}', using bridge_duel")
    DEFAULT_MAP = "bridge_duel"

Arenas = MAP_PRESETS[DEFAULT_MAP]