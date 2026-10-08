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
from typing import Dict, Optional, Tuple

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


def _load_map_dims() -> Dict[int, Tuple[int, int]]:
    """Parse `map_const NAME, W, H ; $ID` into an {id: (width, height)} table.

    The map_const W/H are the map's size in 16x16 *blocks* (e.g. Pallet Town
    is 10x9 blocks), whereas the player x/y we read from RAM are *tiles*. We
    expose tile dimensions (2*W, 2*H) so a per-map visited grid indexes the
    same coordinate space the environment actually observes.
    """
    dims: Dict[int, Tuple[int, int]] = {}
    for path in _GEN1_SOURCES:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = _MAP_CONST_RE.match(line)
                    if m:
                        dims[int(m.group(4), 16)] = (
                            int(m.group(2)) * 2,
                            int(m.group(3)) * 2,
                        )
        except OSError:
            continue
        if dims:
            break
    return dims


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

# Per-map tile dimensions (width, height), keyed by Gen 1 map id. Parsed from
# the pret map constants; converted from block to tile units (see
# _load_map_dims). Missing maps fall back to MAP_DIM_FALLBACK.
MAP_DIMS: Dict[int, Tuple[int, int]] = _load_map_dims()
MAP_DIM_FALLBACK: Tuple[int, int] = (20, 20)

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


def map_dims(map_id: int) -> Tuple[int, int]:
    """Return (width, height) in tiles for a Gen 1 map id.

    Falls back to MAP_DIM_FALLBACK (20, 20) for maps not present in the
    parsed disassembly table so callers always get a usable grid size.
    """
    return MAP_DIMS.get(int(map_id), MAP_DIM_FALLBACK)


def map_progress_by_id(map_id: int) -> float:
    """Return the story-progress fraction (0..1) for a Gen 1 map id.

    Unknown maps (interior rooms, gates, ...) return 0.0 so they neither help
    nor hurt the progress signal.
    """
    name = MAP_NAMES.get(int(map_id))
    if name is None:
        return 0.0
    return MAP_PROGRESS_BY_NAME.get(name, 0.0)


# ---------------------------------------------------------------------------
# Gen 2 (Gold/Silver/Crystal) map knowledge.
#
# Gen 2 identifies maps by a (group, map) byte pair read from wMapGroup /
# wMapNumber. The pret repos (pokegold, pokecrystal) enumerate them with the
# `newgroup` / `map_const` macros; we replicate the assembler's numbering so
# the (group, map) pair we read from RAM maps back to a real map name.
# ---------------------------------------------------------------------------

_GEN2_SOURCES = [
    os.path.join(_REPO_ROOT, "pokegold", "constants", "map_constants.asm"),
    os.path.join(_REPO_ROOT, "pokecrystal", "constants", "map_constants.asm"),
]

_NEWGROUP_RE = re.compile(r"^\s*newgroup\s+(\w+)")
_MAP_CONST2_RE = re.compile(r"^\s*map_const\s+(\w+),\s*\d+,\s*\d+")


def _load_gen2_maps() -> Dict[Tuple[int, int], str]:
    """Parse `newgroup`/`map_const` into {(group, map): NAME} for both repos.

    Group ids start at 1; map ids restart at 1 inside each group (matching
    the macros in pokegold/pokecrystal).
    """
    tables: Dict[Tuple[int, int], str] = {}
    for path in _GEN2_SOURCES:
        if not os.path.exists(path):
            continue
        table: Dict[Tuple[int, int], str] = {}
        group_id = 0
        map_id = 0
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = _NEWGROUP_RE.match(line)
                    if m:
                        group_id += 1
                        map_id = 0
                        continue
                    m2 = _MAP_CONST2_RE.match(line)
                    if m2:
                        map_id += 1
                        table[(group_id, map_id)] = m2.group(1)
        except OSError:
            continue
        if table:
            tables[path] = table
    # Prefer pokecrystal (used by the Crystal ROM) but keep pokegold for
    # Gold/Silver. Both are exposed; callers pick by cartridge title.
    return tables


_GEN2_MAP_TABLES: Dict[str, Dict[Tuple[int, int], str]] = {}
if _GEN2_SOURCES:
    _tables = _load_gen2_maps()
    for _path, _table in _tables.items():
        _key = "crystal" if "pokecrystal" in _path else "gold"
        _GEN2_MAP_TABLES[_key] = _table

# Fallback for the handful of maps that matter before a save even exists.
_GEN2_FALLBACK = {
    (1, 1): "NEW_BARK_TOWN",
    (1, 2): "ELMS_LAB",
    (1, 3): "PLAYERS_HOUSE_1F",
    (1, 4): "PLAYERS_HOUSE_2F",
    (2, 1): "CHERRYGROVE_CITY",
}

Gen2MapTables = Dict[str, Dict[Tuple[int, int], str]]


def gen2_map_table(version: str = "gold") -> Dict[Tuple[int, int], str]:
    """Return the {(group, map): NAME} table for a Gen 2 version."""
    table = (
        _GEN2_MAP_TABLES.get(version)
        or _GEN2_MAP_TABLES.get("gold")
        or dict(_GEN2_FALLBACK)
    )
    if not table:
        table = dict(_GEN2_FALLBACK)
    return table


def gen2_map_name(group: int, number: int, version: str = "gold") -> Optional[str]:
    """Return the map name for a (group, map) pair in a Gen 2 version."""
    return gen2_map_table(version).get((int(group), int(number)))


# ---------------------------------------------------------------------------
# Gen 2 story-progress anchors. Johto (badges 1-8) then Kanto (badges 9-16);
# values are a rough fraction of the *full* two-region game completed by first
# reaching the map (0..1). Same hand-authored approach as Gen 1.
# ---------------------------------------------------------------------------
GEN2_PROGRESS_BY_NAME: Dict[str, float] = {
    # -- Johto intro
    "NEW_BARK_TOWN": 0.00,
    "PLAYERS_HOUSE_1F": 0.00,
    "PLAYERS_HOUSE_2F": 0.00,
    "ELMS_LAB": 0.01,
    "ROUTE_29": 0.03,
    "CHERRYGROVE_CITY": 0.05,
    "ROUTE_30": 0.07,
    "ROUTE_31": 0.08,
    "MR_POKEMONS_HOUSE": 0.09,
    "VIOLET_CITY": 0.12,
    "SPROUT_TOWER_1F": 0.11,
    "VIOLET_GYM": 0.13,  # badge 1
    "ROUTE_32": 0.16,
    "RUINS_OF_ALPH_OUTSIDE": 0.17,
    "UNION_CAVE_1F": 0.19,
    "AZALEA_TOWN": 0.22,
    "SLOWPOKE_WELL_B1F": 0.24,
    "AZALEA_GYM": 0.25,  # badge 2
    "ILEX_FOREST": 0.28,
    "ROUTE_34": 0.31,
    "GOLDENROD_CITY": 0.34,
    "GOLDENROD_GYM": 0.36,  # badge 3
    "RADIO_TOWER_1F": 0.38,
    "ROUTE_35": 0.40,
    "NATIONAL_PARK": 0.42,
    "ROUTE_36": 0.44,
    "ROUTE_37": 0.46,
    "ECRUTEAK_CITY": 0.49,
    "BURNED_TOWER_1F": 0.51,
    "ECRUTEAK_GYM": 0.52,  # badge 4
    "ROUTE_38": 0.55,
    "ROUTE_39": 0.56,
    "OLIVINE_CITY": 0.59,
    "OLIVINE_LIGHTHOUSE_1F": 0.60,
    "ROUTE_40": 0.61,
    "ROUTE_41": 0.62,
    "CIANWOOD_CITY": 0.65,
    "CIANWOOD_GYM": 0.67,  # badge 5 (Chuck; Jasmine absent until return)
    "OLIVINE_GYM": 0.70,  # badge 6 (Jasmine)
    "ROUTE_42": 0.72,
    "MAHOGANY_TOWN": 0.75,
    "TEAM_ROCKET_BASE_B1F": 0.77,
    "MAHOGANY_GYM": 0.79,  # badge 7 (Pryce)
    "ROUTE_43": 0.81,
    "LAKE_OF_RAGE": 0.83,
    "ROUTE_44": 0.85,
    "ICE_PATH_1F": 0.87,
    "BLACKTHORN_CITY": 0.89,
    "BLACKTHORN_GYM_1F": 0.91,  # badge 8 (Clair)
    "DRAGONS_DEN_1F": 0.92,
    "DRAGONS_DEN_B1F": 0.92,
    "ROUTE_45": 0.93,
    "ROUTE_46": 0.93,
    # -- Kanto postgame gate
    "VICTORY_ROAD": 0.96,
    "VICTORY_ROAD_GATE": 0.96,
    "ROUTE_23": 0.97,
    "INDIGO_PLATEAU_POKECENTER_1F": 0.98,
    "WILLS_ROOM": 1.00,
    "KOGAS_ROOM": 1.00,
    "BRUNOS_ROOM": 1.00,
    "KARENS_ROOM": 1.00,
    "LANCES_ROOM": 1.00,
    "CHAMPIONS_ROOM": 1.00,
    "HALL_OF_FAME": 1.00,
}


def gen2_map_progress(group: int, number: int, version: str = "gold") -> float:
    """Story-progress fraction (0..1) for a Gen 2 (group, map) pair."""
    name = gen2_map_name(group, number, version)
    if name is None:
        return 0.0
    return GEN2_PROGRESS_BY_NAME.get(name, 0.0)
