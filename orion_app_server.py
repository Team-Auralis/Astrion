#!/usr/bin/env python3
"""
ORION Mobile App API Server & PWA
Powers the Android Chess App (based on jcarolus/android-chess)
and provides a responsive Mobile Web / PWA Chess App for Android devices.
"""

import http.server
import json
import os
import socketserver
import sys
from urllib.parse import parse_qs, urlparse

from orion.environment.chess import ChessEnvironment
from orion.model.actions import ChessActionSpace
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.search.search_agent import StockfishInspiredSearchAgent
from orion.utils.checkpoint import Checkpoint
from cli import find_best_checkpoint

PORT = 8000
HOST = "0.0.0.0"

network = None
encoder = None
agent = None
checkpoint_loaded = "None"


def init_engine(checkpoint_path=None, depth=2):
    global network, encoder, agent, checkpoint_loaded
    if agent is not None and checkpoint_path is None:
        return agent
    network = ORIONNetwork()
    best_ckpt = find_best_checkpoint(checkpoint_path)
    checkpoint_loaded = "None"
    if best_ckpt and os.path.exists(best_ckpt):
        try:
            ep = Checkpoint().load(network, best_ckpt)
            checkpoint_loaded = f"{best_ckpt} (ep: {ep})"
            print(f"[ORION SERVER] Loaded model: {checkpoint_loaded}")
        except Exception as e:
            print(f"[ORION SERVER] Failed to load {best_ckpt}: {e}")

    encoder = ChessEncoder()
    agent = StockfishInspiredSearchAgent(network, encoder, default_depth=depth)
    return agent


HTML_MOBILE_APP = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <meta name="theme-color" content="#1a1c23">
  <title>ORION Chess AI</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #12141a;
      color: #e2e8f0;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 12px;
    }
    header {
      width: 100%;
      max-width: 480px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 0;
      border-bottom: 1px solid #2d3748;
      margin-bottom: 12px;
    }
    h1 { font-size: 1.25rem; color: #60a5fa; }
    .badge { font-size: 0.75rem; background: #2563eb; color: #fff; padding: 2px 8px; border-radius: 9999px; }
    .status-bar {
      width: 100%;
      max-width: 480px;
      background: #1e293b;
      padding: 10px;
      border-radius: 8px;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      font-size: 0.9rem;
    }
    #board {
      width: 100%;
      max-width: 440px;
      aspect-ratio: 1 / 1;
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      grid-template-rows: repeat(8, 1fr);
      border: 3px solid #334155;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }
    .square {
      display: flex;
      justify-content: center;
      align-items: center;
      font-size: clamp(2rem, 7vw, 2.75rem);
      cursor: pointer;
      position: relative;
    }
    .light { background: #e2d6b5; color: #222; }
    .dark { background: #b88b4a; color: #222; }
    .selected { background: #f6e05e !important; }
    .highlight { background: rgba(59, 130, 246, 0.5) !important; }
    .last-move { background: rgba(74, 222, 128, 0.4) !important; }
    .controls {
      width: 100%;
      max-width: 480px;
      margin-top: 16px;
      display: flex;
      gap: 10px;
    }
    button {
      flex: 1;
      background: #3b82f6;
      color: white;
      border: none;
      padding: 12px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 1rem;
      cursor: pointer;
      touch-action: manipulation;
    }
    button:active { background: #1d4ed8; }
    .depth-picker {
      width: 100%;
      max-width: 480px;
      margin-top: 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.85rem;
      color: #94a3b8;
    }
    select {
      background: #1e293b;
      color: #e2e8f0;
      border: 1px solid #334155;
      padding: 6px 12px;
      border-radius: 6px;
    }
  </style>
</head>
<body>
  <header>
    <h1>🌌 ORION Mobile Chess</h1>
    <span class="badge">Team-Auralis</span>
  </header>

  <div class="status-bar">
    <span id="turn-indicator">Your Turn (White)</span>
    <span id="eval-indicator">Eval: 0.0</span>
  </div>

  <div id="board"></div>

  <div class="depth-picker">
    <label>AI Strength:</label>
    <select id="depth-select">
      <option value="1">Depth 1 (Fast)</option>
      <option value="2" selected>Depth 2 (Standard - 1200 Elo)</option>
      <option value="3">Depth 3 (Tactical Master - 1350+ Elo)</option>
    </select>
  </div>

  <div class="controls">
    <button onclick="newGame()">New Game</button>
    <button onclick="aiMove()" style="background:#10b981;">AI Move</button>
  </div>

  <script>
    const PIECE_SYMBOLS = {
      'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
      'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
      '.': ''
    };

    let board = [
      ["r","n","b","q","k","b","n","r"],
      ["p","p","p","p","p","p","p","p"],
      [".",".",".",".",".",".",".","."],
      [".",".",".",".",".",".",".","."],
      [".",".",".",".",".",".",".","."],
      [".",".",".",".",".",".",".","."],
      ["P","P","P","P","P","P","P","P"],
      ["R","N","B","Q","K","B","N","R"]
    ];

    let turn = 'white';
    let selectedSquare = null;
    let lastFrom = null, lastTo = null;

    function renderBoard() {
      const boardEl = document.getElementById('board');
      boardEl.innerHTML = '';
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const sq = document.createElement('div');
          const isLight = (r + c) % 2 === 0;
          sq.className = `square ${isLight ? 'light' : 'dark'}`;
          sq.dataset.r = r;
          sq.dataset.c = c;

          if (selectedSquare && selectedSquare.r === r && selectedSquare.c === c) {
            sq.classList.add('selected');
          }
          if ((lastFrom && lastFrom.r === r && lastFrom.c === c) ||
              (lastTo && lastTo.r === r && lastTo.c === c)) {
            sq.classList.add('last-move');
          }

          const piece = board[r][c];
          sq.textContent = PIECE_SYMBOLS[piece] || '';
          sq.onclick = () => onSquareClick(r, c);
          boardEl.appendChild(sq);
        }
      }
      document.getElementById('turn-indicator').textContent =
        turn === 'white' ? 'Your Turn (White)' : 'ORION Thinking (Black)...';
    }

    function onSquareClick(r, c) {
      if (turn !== 'white') return;
      const piece = board[r][c];

      if (selectedSquare) {
        if (selectedSquare.r === r && selectedSquare.c === c) {
          selectedSquare = null;
          renderBoard();
          return;
        }
        // Execute move
        const uciMove = coordsToUci(selectedSquare.r, selectedSquare.c, r, c);
        makeMoveLocally(selectedSquare.r, selectedSquare.c, r, c);
        selectedSquare = null;
        renderBoard();
        // Request ORION AI response
        setTimeout(aiMove, 300);
      } else {
        if (piece !== '.' && piece === piece.toUpperCase()) {
          selectedSquare = { r, c };
          renderBoard();
        }
      }
    }

    function coordsToUci(r1, c1, r2, c2) {
      const files = "abcdefgh";
      return files[c1] + (8 - r1) + files[c2] + (8 - r2);
    }

    function uciToCoords(uci) {
      const files = "abcdefgh";
      return {
        r1: 8 - parseInt(uci[1]),
        c1: files.indexOf(uci[0]),
        r2: 8 - parseInt(uci[3]),
        c2: files.indexOf(uci[2])
      };
    }

    function makeMoveLocally(r1, c1, r2, c2) {
      const piece = board[r1][c1];
      board[r2][c2] = piece;
      board[r1][c1] = '.';
      // Simple pawn promotion
      if (piece === 'P' && r2 === 0) board[r2][c2] = 'Q';
      if (piece === 'p' && r2 === 7) board[r2][c2] = 'q';
      lastFrom = { r: r1, c: c1 };
      lastTo = { r: r2, c: c2 };
      turn = (turn === 'white') ? 'black' : 'white';
    }

    function boardToFen() {
      let fen = "";
      for (let r = 0; r < 8; r++) {
        let empty = 0;
        for (let c = 0; c < 8; c++) {
          let p = board[r][c];
          if (p === '.') {
            empty++;
          } else {
            if (empty > 0) { fen += empty; empty = 0; }
            fen += p;
          }
        }
        if (empty > 0) fen += empty;
        if (r < 7) fen += "/";
      }
      fen += ` ${turn === 'white' ? 'w' : 'b'} KQkq - 0 1`;
      return fen;
    }

    async function aiMove() {
      const fen = boardToFen();
      const depth = parseInt(document.getElementById('depth-select').value);
      document.getElementById('turn-indicator').textContent = 'ORION Thinking...';

      try {
        const resp = await fetch('/api/move', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fen: fen, depth: depth })
        });
        const data = await resp.json();
        if (data.best_move && data.best_move.length >= 4) {
          const m = uciToCoords(data.best_move);
          makeMoveLocally(m.r1, m.c1, m.r2, m.c2);
          document.getElementById('eval-indicator').textContent = `Eval: ${(data.score / 100).toFixed(2)} (${data.nodes} nodes)`;
        }
      } catch (err) {
        console.error("AI Move error:", err);
      }
      renderBoard();
    }

    function newGame() {
      board = [
        ["r","n","b","q","k","b","n","r"],
        ["p","p","p","p","p","p","p","p"],
        [".",".",".",".",".",".",".","."],
        [".",".",".",".",".",".",".","."],
        [".",".",".",".",".",".",".","."],
        [".",".",".",".",".",".",".","."],
        ["P","P","P","P","P","P","P","P"],
        ["R","N","B","Q","K","B","N","R"]
      ];
      turn = 'white';
      selectedSquare = null;
      lastFrom = null; lastTo = null;
      document.getElementById('eval-indicator').textContent = 'Eval: 0.0';
      renderBoard();
    }

    renderBoard();
  </script>
</body>
</html>
"""


class OrionServerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            if agent is None:
                init_engine()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = {
                "name": "ORION Gaming Mind",
                "version": "0.6",
                "author": "Team-Auralis",
                "checkpoint": checkpoint_loaded,
                "status": "online",
                "default_depth": agent.default_depth if agent else 2
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))
        elif parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_MOBILE_APP.encode("utf-8"))
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/move":
            if agent is None:
                init_engine()
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(body)
            except Exception:
                payload = {}

            fen = payload.get("fen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            depth = int(payload.get("depth", 2))

            env = ChessEnvironment()
            try:
                env.load_fen(fen)
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Invalid FEN: {e}"}).encode("utf-8"))
                return

            best_move = agent.choose_move(env, depth=depth)
            score = agent.evaluate_board(env)
            best_uci = ChessActionSpace.move_to_uci(best_move) if best_move else "0000"

            response_data = {
                "best_move": best_uci,
                "from": best_uci[:2] if len(best_uci) >= 4 else "",
                "to": best_uci[2:4] if len(best_uci) >= 4 else "",
                "score": score,
                "nodes": agent.nodes_evaluated,
                "fen": fen
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_server(host="0.0.0.0", port=8000, checkpoint=None, depth=2):
    global PORT, HOST
    PORT = port
    HOST = host
    init_engine(checkpoint, depth)

    print("================================================================")
    print("           ORION MOBILE APP SERVER & BACKEND                    ")
    print("================================================================")
    print(f"Host        : {host}")
    print(f"Server Port : {port}")
    print(f"Model       : {checkpoint_loaded}")
    print(f"URL (Local) : http://localhost:{port}")
    print(f"API Endpoint: http://localhost:{port}/api/move")
    print("================================================================\n")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, port), OrionServerHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down ORION Server.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ORION Mobile App Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--depth", type=int, default=2, help="Search depth")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, checkpoint=args.checkpoint, depth=args.depth)


if __name__ == "__main__":
    main()

