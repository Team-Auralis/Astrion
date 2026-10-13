from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder

game = ChessEnvironment()
encoder = ChessEncoder()

state = game.get_state()

tensor = encoder.encode(state)
flat = encoder.flatten(tensor)

print("================================")
print("       ORION STATE ENCODER      ")
print("================================")

print("Tensor channels :", len(tensor))
print("Rows             :", len(tensor[0]))
print("Columns          :", len(tensor[0][0]))
print("Flattened size   :", len(flat))
print("Active values    :", encoder.count_active(tensor))

assert len(tensor) == 18
assert len(tensor[0]) == 8
assert len(tensor[0][0]) == 8
assert len(flat) == 18 * 8 * 8

# Starting position:
# 32 pieces + 64 squares side-to-move + 4*64 squares castling rights
# = 352 active values.
assert encoder.count_active(tensor) == 352

print()
print("[PASS] Tensor shape")
print("[PASS] Flattening")
print("[PASS] Starting position encoding")
print()
print("STATE ENCODER READY")
