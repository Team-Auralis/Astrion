"""
ORION Pokemon Gen 1/2 AI
========================
Adapts ORION's ResNet architecture for Pokemon Red/Blue/Yellow gameplay.
Uses PyBoy Game Boy emulator for Gen 1, with Gen 2 (Gold/Silver/Crystal) support.
"""

from orion.pokemon.environment import PokemonEnv, NUM_ACTIONS, BUTTONS
from orion.pokemon.network import PokemonResNet, PokemonEncoder
from orion.pokemon.gen2_env import PokemonGen2Env

__all__ = [
    "PokemonEnv",
    "PokemonGen2Env",
    "PokemonResNet",
    "PokemonEncoder",
    "NUM_ACTIONS",
    "BUTTONS",
]
