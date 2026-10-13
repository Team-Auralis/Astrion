from orion.environment.chess import ChessEnvironment

game = ChessEnvironment()

print("================================")
print("       ORION GAME AI v0.4       ")
print("       Chess Environment        ")
print("================================")

game.display()

legal = game.legal_moves()
status = game.game_status()

print()
print("Legal moves :", len(legal))
print("Game status :", status)
print("Turn        :", game.turn)
print("Move count  :", game.move_count)

assert len(legal) == 20
assert status == "playing"
assert game.turn == "white"
assert game.move_count == 0

print()
print("[PASS] Environment initialized")
print("[PASS] Starting position has 20 legal moves")
print("[PASS] Initial game state is valid")
print()
print("ORION v0.4 ENVIRONMENT ONLINE")
