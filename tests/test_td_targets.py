from orion.training.train import ORIONTrainer


trainer = ORIONTrainer(seed=42)


# --------------------------------------------------
# NON-TERMINAL TD TARGETS
# --------------------------------------------------

target = trainer.compute_value_target(
    player="white",
    winner=None,
    next_value=0.20
)

assert abs(target - 0.80) < 1e-12


target = trainer.compute_value_target(
    player="white",
    winner=None,
    next_value=0.50
)

assert abs(target - 0.50) < 1e-12


target = trainer.compute_value_target(
    player="black",
    winner=None,
    next_value=0.90
)

assert abs(target - 0.10) < 1e-12


# --------------------------------------------------
# TERMINAL RESULTS
# --------------------------------------------------

target = trainer.compute_value_target(
    player="white",
    winner="white",
    next_value=None
)

assert abs(target - 1.0) < 1e-12


target = trainer.compute_value_target(
    player="white",
    winner="black",
    next_value=None
)

assert abs(target - 0.0) < 1e-12


target = trainer.compute_value_target(
    player="black",
    winner=None,
    next_value=None
)

assert abs(target - 0.5) < 1e-12


print("================================")
print("     ORION v0.5 TD TARGET TEST")
print("================================")
print()
print("0.20 ->", trainer.compute_value_target("white", None, 0.20))
print("0.50 ->", trainer.compute_value_target("white", None, 0.50))
print("0.90 ->", trainer.compute_value_target("black", None, 0.90))
print("Win   ->", trainer.compute_value_target("white", "white", None))
print("Loss  ->", trainer.compute_value_target("white", "black", None))
print("Draw  ->", trainer.compute_value_target("black", None, None))
print()
print("[PASS] Non-terminal TD inversion")
print("[PASS] Terminal win target")
print("[PASS] Terminal loss target")
print("[PASS] Terminal draw target")
print()
print("ORION v0.5 TD TARGETS VERIFIED")
