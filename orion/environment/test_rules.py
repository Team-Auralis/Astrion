from orion.environment.chess import ChessEnvironment

game = ChessEnvironment()

print("ORION CHESS RULE TEST")
print("=====================")

# Starting position
moves = game.legal_moves()

assert len(moves) == 20, (
    f"Expected 20 opening moves, got {len(moves)}"
)

print("[PASS] Starting position has 20 moves")

# e2-e4
assert game.make_move((6, 4, 4, 4))
print("[PASS] e2-e4")

# Black should now have 20 moves
moves = game.legal_moves()

assert len(moves) == 20, (
    f"Expected 20 black responses, got {len(moves)}"
)

print("[PASS] Black has 20 responses")

# Invalid move: white pawn moving backwards
assert not game.make_move((4, 4, 5, 4))
print("[PASS] Illegal pawn move rejected")

# Reset
game.reset()

assert game.turn == "white"
assert game.move_count == 0

print("[PASS] Reset works")

print()
print("ALL BASIC TESTS PASSED")
