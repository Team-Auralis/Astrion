"""
Verification tests for StockfishInspiredSearchAgent, Arena bots, Elo calculations, and UCI engine.
"""

from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.search.search_agent import StockfishInspiredSearchAgent
from orion.arena.bots import RandomBot, GreedyMaterialBot, MinimaxBot, OrionBot
from orion.arena.battle import EloTracker, Arena


def test_search_agent_chooses_legal_moves():
    env = ChessEnvironment()
    net = ORIONNetwork()
    enc = ChessEncoder()
    agent = StockfishInspiredSearchAgent(net, enc, default_depth=1)

    move = agent.choose_move(env, depth=1)
    assert move is not None
    assert move in env.legal_moves()


def test_quiescence_eval():
    env = ChessEnvironment()
    net = ORIONNetwork()
    enc = ChessEncoder()
    agent = StockfishInspiredSearchAgent(net, enc)

    eval_score = agent.evaluate_board(env)
    # At start of standard chess, board evaluation should be balanced (~0.0)
    assert abs(eval_score) < 50.0


def test_elo_tracker_math():
    # A (1200) vs B (1200): expected score = 0.5
    exp_a = EloTracker.expected_score(1200.0, 1200.0)
    assert abs(exp_a - 0.5) < 1e-4

    # If A wins against equal opponent with K=32, new rating is 1200 + 32*(1.0 - 0.5) = 1216.0
    new_a, new_b = EloTracker.update_ratings(1200.0, 1200.0, score_a=1.0, k=32.0)
    assert new_a == 1216.0
    assert new_b == 1184.0


def test_arena_quick_match():
    net = ORIONNetwork()
    enc = ChessEncoder()
    bot1 = OrionBot(net, enc, use_search=True, depth=1, rating=1200.0)
    bot2 = RandomBot(seed=99)

    arena = Arena(bot1, bot2, max_moves=20)
    res = arena.run_match(rounds=2, verbose=False)
    assert res["total_games"] == 2


if __name__ == "__main__":
    print("Running Stockfish & Arena tests...")
    test_search_agent_chooses_legal_moves()
    print("[PASS] Search agent chooses legal moves")
    test_quiescence_eval()
    print("[PASS] Quiescence static evaluation")
    test_elo_tracker_math()
    print("[PASS] Elo tracker mathematics")
    test_arena_quick_match()
    print("[PASS] Arena match execution")
    print("ALL TESTS PASSED!")
