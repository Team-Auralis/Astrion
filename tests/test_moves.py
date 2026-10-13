from orion.environment.chess import ChessEnvironment

game = ChessEnvironment()

print("ORION — Legal Move Test")
game.display()

moves = game.legal_moves()

print("Legal moves:", len(moves))
print()

for move in moves:
    print(game.move_to_text(move))

print("\nTesting e2-e4...")

success = game.make_move((6,4,4,4))

print("Move successful:", success)
game.display()
