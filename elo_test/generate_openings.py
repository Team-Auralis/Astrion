"""
Generate balanced opening suite for ORION tournament evaluation.
Standard openings 2-4 plies deep to ensure diverse, fair opening positions.
"""
import os

openings = [
    # 1. Open Game (e4 e5)
    ("Ruy Lopez", "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *"),
    ("Italian Game", "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 *"),
    ("Four Knights", "1. e4 e5 2. Nf3 Nc6 3. Nc3 Nf6 *"),
    ("Scotch Game", "1. e4 e5 2. Nf3 Nc6 3. d4 exd4 *"),
    ("Petrov Defense", "1. e4 e5 2. Nf3 Nf6 *"),
    
    # 2. Semi-Open (e4 ...)
    ("Sicilian Open", "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 *"),
    ("Sicilian Classical", "1. e4 c5 2. Nf3 Nc6 3. d4 cxd4 *"),
    ("French Winawer", "1. e4 e6 2. d4 d5 3. Nc3 Bb4 *"),
    ("French Classical", "1. e4 e6 2. d4 d5 3. Nc3 Nf6 *"),
    ("Caro-Kann Advance", "1. e4 c6 2. d4 d5 3. e5 Bf5 *"),
    ("Caro-Kann Classical", "1. e4 c6 2. d4 d5 3. Nc3 dxe4 *"),
    ("Pirc Defense", "1. e4 d6 2. d4 Nf6 3. Nc3 g6 *"),
    
    # 3. Closed Game (d4 d5)
    ("Queen's Gambit Declined", "1. d4 d5 2. c4 e6 3. Nc3 Nf6 *"),
    ("Queen's Gambit Accepted", "1. d4 d5 2. c4 dxc4 3. Nf3 *"),
    ("Slav Defense", "1. d4 d5 2. c4 c6 3. Nf3 Nf6 *"),
    
    # 4. Indian Defenses (d4 Nf6)
    ("King's Indian Defense", "1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 *"),
    ("Nimzo-Indian Defense", "1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 *"),
    ("Queen's Indian Defense", "1. d4 Nf6 2. c4 e6 3. Nf3 b6 *"),
    ("Grunfeld Defense", "1. d4 Nf6 2. c4 g6 3. Nc3 d5 *"),
    
    # 5. Flank Openings (c4 / Nf3)
    ("English Opening", "1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6 *"),
    ("Reti Opening", "1. Nf3 d5 2. c4 c6 3. g3 Nf6 *"),
    ("King's English", "1. c4 c5 2. Nc3 Nc6 *"),
]

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openings.pgn")
with open(out_path, "w", encoding="utf-8") as f:
    for i, (eco_name, moves) in enumerate(openings, 1):
        f.write(f'[Event "Opening Suite"]\n')
        f.write(f'[Site "Orion Validation"]\n')
        f.write(f'[Date "2026.10.10"]\n')
        f.write(f'[Round "{i}"]\n')
        f.write(f'[White "?"]\n')
        f.write(f'[Black "?"]\n')
        f.write(f'[Result "*"]\n')
        f.write(f'[Opening "{eco_name}"]\n\n')
        f.write(f'{moves}\n\n')

print(f"Generated {len(openings)} openings in {out_path}")
