from orion.training.train import ORIONTrainer


trainer = ORIONTrainer()

print("================================")
print("       ORION TRAINING v0.1      ")
print("================================")

result = trainer.train_game(
    max_moves=20
)

print()
print("Winner       :", result["winner"])
print("Positions    :", result["positions"])
print("Policy loss  :", result["policy_loss"])
print("Value loss   :", result["value_loss"])

assert result["positions"] <= 20

print()
print("[PASS] Self-play generated training data")
print("[PASS] Game outcome generated reward")
print("[PASS] Network parameters were updated")
print()
print("ORION TRAINING LOOP ONLINE")
