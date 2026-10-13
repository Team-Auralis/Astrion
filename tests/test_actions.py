from orion.environment.chess import ChessEnvironment
from orion.model.actions import ChessActionSpace

game = ChessEnvironment()

print("================================")
print("       ORION ACTION SPACE       ")
print("================================")

moves = game.legal_moves()

print("Legal moves:", len(moves))
print("Action space:", ChessActionSpace.SIZE)

for move in moves[:5]:

    action = ChessActionSpace.encode(move)
    decoded = ChessActionSpace.decode(action)

    print(
        ChessActionSpace.move_to_uci(move),
        "->",
        action,
        "->",
        ChessActionSpace.move_to_uci(decoded)
    )

mask = ChessActionSpace.legal_action_mask(game)

print()
print("Mask size:", len(mask))
print("Legal actions in mask:", sum(mask))

assert len(mask) == 4096
assert sum(mask) == 20

for move in moves:

    action = ChessActionSpace.encode(move)

    assert mask[action] == 1
    assert ChessActionSpace.decode(action) == move

print()
print("[PASS] Action encoding")
print("[PASS] Action decoding")
print("[PASS] Legal-action mask")
print()
print("ACTION SPACE READY")
