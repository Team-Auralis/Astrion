"""Quick smoke test: verify PyBoy environment works with real ROM."""

from orion.pokemon.environment import PokemonEnv

env = PokemonEnv(
    rom_path=r"D:\roms\Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb",
    headless=True,
    max_steps=100,
)
print("Init OK - boot state saved")

obs, info = env.reset()
print(f"Reset OK - screen: {obs['screen'].shape}, state: {obs['state'].shape}")
print(f"Screen range: {obs['screen'].min()}-{obs['screen'].max()}")

for i in range(10):
    obs, reward, term, trunc, info = env.step(i % 8)
print(f"10 steps OK - reward: {reward:.2f}, badges: {info['badges']}")

obs, info = env.reset()
print(f"Second reset OK - screen: {obs['screen'].shape}")

env.close()
print("ALL TESTS PASSED")
