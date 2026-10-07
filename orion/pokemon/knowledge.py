"""Game knowledge extracted from the pret disassembly projects.

The pret project (https://pret.github.io/) publishes full disassemblies of the
Game Boy Pokemon games. We mine them for structured knowledge that the RL
reward function can use -- starting with the Gen 1 map table, which lets us
score "how far into the story" a map is.

Nothing here is hard-required: if the disassembly is missing we fall back to a
small embedded table so training still runs.
"""

import os
import re
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Locate the disassembly repos (cloned next to the project by the operator).
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

_GEN1_SOURCES = [
    os.path.join(_REPO_ROOT, "pokered", "constants", "map_constants.asm"),
    os.path.join(_REPO_ROOT, "pokeyellow", "constants", "map_constants.asm"),
]

_MAP_CONST_RE = re.compile(
    r"^\s*map_const\s+(\w+),\s*(\d+),\s*(\d+)\s*;\s*\$([0-9A-Fa-f]+)"
)


def _load_map_names() -> Dict[int, str]:
    """Parse `map_const NAME, W, H ; $ID` lines into an {id: name} table."""
    names: Dict[int, str] = {}
    for path in _GEN1_SOURCES:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = _MAP_CONST_RE.match(line)
                    if m:
                        names[int(m.group(4), 16)] = m.group(1)
        except OSError:
            continue
        if names:
            break
    return names


# Minimum fallback so the module is usable without the disassembly.
_FALLBACK_NAMES = {
    0x00: "PALLET_TOWN",
    0x01: "VIRIDIAN_CITY",
    0x02: "PEWTER_CITY",
    0x03: "CERULEAN_CITY",
    0x25: "REDS_HOUSE_1F",
    0x26: "REDS_HOUSE_2F",
    0x28: "OAKS_LAB",
    0x33: "VIRIDIAN_FOREST",
    0x36: "PEWTER_GYM",
}

MAP_NAMES: Dict[int, str] = _load_map_names() or dict(_FALLBACK_NAMES)

# ---------------------------------------------------------------------------
# Story-progress anchors. Values are a rough fraction (0..1) of the main
# game completed by first *reaching* the map. Hand-authored from the intended
# progression order (gyms 1..8, then Victory Road -> Indigo Plateau).
# ---------------------------------------------------------------------------
MAP_PROGRESS_BY_NAME: Dict[str, float] = {
    "PALLET_TOWN": 0.00,
    "REDS_HOUSE_2F": 0.00,
    "REDS_HOUSE_1F": 0.00,
    "OAKS_LAB": 0.02,
    "VIRIDIAN_CITY": 0.04,
    "ROUTE_1": 0.06,
    "ROUTE_2": 0.07,
    "VIRIDIAN_FOREST": 0.10,
    "PEWTER_CITY": 0.13,
    "PEWTER_GYM": 0.13,
    "ROUTE_3": 0.16,
    "MT_MOON_1F": 0.19,
    "MT_MOON_B1F": 0.20,
    "MT_MOON_B2F": 0.20,
    "ROUTE_4": 0.22,
    "CERULEAN_CITY": 0.25,
    "CERULEAN_GYM": 0.26,
    "ROUTE_24": 0.28,
    "ROUTE_25": 0.29,
    "BILLS_HOUSE": 0.29,
    "ROUTE_5": 0.31,
    "ROUTE_6": 0.32,
    "VERMILION_CITY": 0.35,
    "VERMILION_GYM": 0.36,
    "SS_ANNE_1F": 0.38,
    "SS_ANNE_BOW": 0.39,
    "ROUTE_9": 0.42,
    "ROUTE_10": 0.43,
    "ROCK_TUNNEL_1F": 0.46,
    "ROCK_TUNNEL_B1F": 0.47,
    "LAVENDER_TOWN": 0.50,
    "POKEMON_TOWER_1F": 0.52,
    "POKEMON_TOWER_7F": 0.55,
    "ROUTE_8": 0.57,
    "ROUTE_7": 0.58,
    "CELADON_CITY": 0.61,
    "CELADON_GYM": 0.62,
    "ROCKET_HIDEOUT_B1F": 0.64,
    "ROCKET_HIDEOUT_B4F": 0.66,
    "SAFFRON_CITY": 0.70,
    "SILPH_CO_1F": 0.72,
    "SILPH_CO_7F": 0.75,
    "SILPH_CO_11F": 0.76,
    "SAFFRON_GYM": 0.78,
    "FUCHSIA_CITY": 0.80,
    "FUCHSIA_GYM": 0.81,
    "SAFARI_ZONE_GATE": 0.82,
    "SEAFOAM_ISLANDS_1F": 0.84,
    "CINNABAR_ISLAND": 0.86,
    "CINNABAR_GYM": 0.87,
    "POKEMON_MANSION_1F": 0.88,
    "VIRIDIAN_GYM": 0.92,
    "ROUTE_22": 0.94,
    "ROUTE_23": 0.95,
    "VICTORY_ROAD_1F": 0.97,
    "VICTORY_ROAD_2F": 0.97,
    "VICTORY_ROAD_3F": 0.98,
    "INDIGO_PLATEAU": 0.99,
    "INDIGO_PLATEAU_LOBBY": 0.99,
    "LORELEIS_ROOM": 1.00,
    "BRUNOS_ROOM": 1.00,
    "AGATHAS_ROOM": 1.00,
    "LANCES_ROOM": 1.00,
    "CHAMPIONS_ROOM": 1.00,
    "HALL_OF_FAME": 1.00,
}


def map_name(map_id: int) -> Optional[str]:
    """Return the human-readable name for a Gen 1 map id, if known."""
    return MAP_NAMES.get(int(map_id))


def map_progress_by_id(map_id: int) -> float:
    """Return the story-progress fraction (0..1) for a Gen 1 map id.

    Unknown maps (interior rooms, gates, ...) return 0.0 so they neither help
    nor hurt the progress signal.
    """
    name = MAP_NAMES.get(int(map_id))
    if name is None:
        return 0.0
    return MAP_PROGRESS_BY_NAME.get(name, 0.0)
