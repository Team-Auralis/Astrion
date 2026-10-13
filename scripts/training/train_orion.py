from orion.training.session import TrainingSession


print("================================")
print("      ORION TRAINING v0.2       ")
print("================================")
print()

session = TrainingSession()

results = session.run(
    episodes=10,
    max_moves=100
)

print()
print("================================")
print("          RESULTS               ")
print("================================")

print("Wins   :", session.wins)
print("Draws  :", session.draws)
print("Losses :", session.losses)

print()
print("Training complete.")
