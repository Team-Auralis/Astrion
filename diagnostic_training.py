from collections import Counter

from orion.training.train import ORIONTrainer


EPISODES = 50
MAX_MOVES = 100


trainer = ORIONTrainer(
    seed=42,
    learning_rate=0.0001,
    temperature=1.0
)

results = []

opening_moves = Counter()

for episode in range(
    1,
    EPISODES + 1
):

    result = trainer.train_game(
        max_moves=MAX_MOVES
    )

    results.append(result)

    print(
        f"Game {episode:03d} | "
        f"winner={result['winner'] or 'draw':5s} | "
        f"moves={result['positions']:3d} | "
        f"policy={result['policy_loss']:.4f} | "
        f"value={result['value_loss']:.4f}"
    )

    # Re-create a fresh game only to inspect
    # the current opening preference.
    from orion.environment.chess import ChessEnvironment

    game = ChessEnvironment()

    move = trainer.agent.choose_move(
        game,
        temperature=1.0
    )

    if move is not None:
        opening_moves[move] += 1


wins = sum(
    1
    for r in results
    if r["winner"] == "white"
)

losses = sum(
    1
    for r in results
    if r["winner"] == "black"
)

draws = EPISODES - wins - losses

average_moves = (
    sum(r["positions"] for r in results)
    / EPISODES
)

average_policy = (
    sum(r["policy_loss"] for r in results)
    / EPISODES
)

average_value = (
    sum(r["value_loss"] for r in results)
    / EPISODES
)

print()
print("================================")
print("   ORION v0.3 DIAGNOSTIC RUN    ")
print("================================")

print("Games          :", EPISODES)
print("White wins     :", wins)
print("Black wins     :", losses)
print("Draws          :", draws)
print(
    "Average moves  :",
    round(average_moves, 2)
)
print(
    "Average policy :",
    round(average_policy, 6)
)
print(
    "Average value  :",
    round(average_value, 6)
)

print()
print("Opening move distribution:")

for move, count in opening_moves.most_common():
    print(
        f"{move}: {count}"
    )

print()
print("DIAGNOSTIC COMPLETE")
