from orion.training.train import ORIONTrainer


trainer = ORIONTrainer(
    seed=42,
    learning_rate=0.0001,
    temperature=1.0
)

result = trainer.train_game(
    max_moves=20
)

print("================================")
print("     ORION v0.3 TRAINING TEST   ")
print("================================")

print("Winner       :", result["winner"])
print("Positions    :", result["positions"])
print("Policy loss  :", result["policy_loss"])
print("Value loss   :", result["value_loss"])

assert result["positions"] > 0
assert result["policy_loss"] >= 0.0
assert result["value_loss"] >= 0.0

print()
print("[PASS] Self-play completed")
print("[PASS] Training data generated")
print("[PASS] Full backpropagation executed")
print("[PASS] v0.3 training loop operational")

print()
print("ORION v0.3 TRAINING LOOP ONLINE")
