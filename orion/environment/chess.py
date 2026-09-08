class ChessEnvironment:

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [
            ["r","n","b","q","k","b","n","r"],
            ["p","p","p","p","p","p","p","p"],
            [".",".",".",".",".",".",".","."],
            [".",".",".",".",".",".",".","."],
            [".",".",".",".",".",".",".","."],
            [".",".",".",".",".",".",".","."],
            ["P","P","P","P","P","P","P","P"],
            ["R","N","B","Q","K","B","N","R"]
        ]

        self.turn = "white"
        self.move_count = 0

        self.castling = {
            "K": True,
            "Q": True,
            "k": True,
            "q": True
        }

        self.en_passant = None

        # Number of half-moves since the last pawn move
        # or capture. 100 half-moves = 50-move rule.
        self.halfmove_clock = 0

        # Position occurrence history for threefold repetition.
        self.position_history = {}

        self._record_position()

        return self.get_state()

    def load_fen(self, fen):
        """
        Load a chess position from a standard FEN string.

        FEN fields:
        1. piece placement
        2. active color
        3. castling availability
        4. en-passant target square
        5. halfmove clock
        6. fullmove number
        """

        fields = fen.strip().split()

        if len(fields) != 6:
            raise ValueError(
                "FEN must contain exactly 6 fields."
            )

        board_part = fields[0]
        active_color = fields[1]
        castling_part = fields[2]
        en_passant_part = fields[3]
        halfmove_part = fields[4]
        fullmove_part = fields[5]

        rows = board_part.split("/")

        if len(rows) != 8:
            raise ValueError(
                "FEN board must contain 8 ranks."
            )

        board = []

        for rank in rows:

            row = []

            for char in rank:

                if char.isdigit():

                    count = int(char)

                    if count < 1 or count > 8:
                        raise ValueError(
                            "Invalid FEN empty-square count."
                        )

                    row.extend(["."] * count)

                elif char in "prnbqkPRNBQK":

                    row.append(char)

                else:

                    raise ValueError(
                        f"Invalid FEN piece: {char}"
                    )

            if len(row) != 8:
                raise ValueError(
                    "Each FEN rank must contain 8 squares."
                )

            board.append(row)

        if active_color == "w":
            turn = "white"
        elif active_color == "b":
            turn = "black"
        else:
            raise ValueError(
                "Invalid FEN active color."
            )

        castling = {
            "K": "K" in castling_part,
            "Q": "Q" in castling_part,
            "k": "k" in castling_part,
            "q": "q" in castling_part
        }

        if castling_part != "-" and any(
            char not in "KQkq"
            for char in castling_part
        ):
            raise ValueError(
                "Invalid FEN castling field."
            )

        if en_passant_part == "-":

            en_passant = None

        else:

            if (
                len(en_passant_part) != 2
                or en_passant_part[0] not in "abcdefgh"
                or en_passant_part[1] not in "12345678"
            ):
                raise ValueError(
                    "Invalid FEN en-passant square."
                )

            en_passant = (
                8 - int(en_passant_part[1]),
                ord(en_passant_part[0]) - ord("a")
            )

        try:
            halfmove_clock = int(halfmove_part)
            fullmove_number = int(fullmove_part)
        except ValueError:
            raise ValueError(
                "Invalid FEN move counters."
            )

        if halfmove_clock < 0:
            raise ValueError(
                "FEN halfmove clock cannot be negative."
            )

        if fullmove_number < 1:
            raise ValueError(
                "FEN fullmove number must be at least 1."
            )

        self.board = board
        self.turn = turn
        self.castling = castling
        self.en_passant = en_passant
        self.halfmove_clock = halfmove_clock

        # Internally move_count represents plies.
        # FEN's fullmove number starts at 1.
        if turn == "white":
            self.move_count = (
                fullmove_number - 1
            ) * 2
        else:
            self.move_count = (
                fullmove_number - 1
            ) * 2 + 1

        self.position_history = {}
        self._record_position()

        return self.get_state()

    def _position_key(self):
        """
        Returns the information that defines a chess position
        for repetition detection.

        Board + side to move + castling rights + en-passant target.
        """

        board_key = tuple(
            piece
            for row in self.board
            for piece in row
        )

        castling_key = tuple(
            key
            for key in ("K", "Q", "k", "q")
            if self.castling[key]
        )

        return (
            board_key,
            self.turn,
            castling_key,
            self.en_passant
        )

    def _record_position(self):
        key = self._position_key()

        self.position_history[key] = (
            self.position_history.get(key, 0) + 1
        )

    def repetition_count(self):
        return self.position_history.get(
            self._position_key(),
            0
        )

    def get_state(self):
        return {
            "board": [row[:] for row in self.board],
            "turn": self.turn,
            "move_count": self.move_count,
            "halfmove_clock": self.halfmove_clock,
            "castling": self.castling.copy(),
            "en_passant": self.en_passant
        }

    def inside(self, r, c):
        return 0 <= r < 8 and 0 <= c < 8

    def is_white(self, piece):
        return piece != "." and piece.isupper()

    def is_black(self, piece):
        return piece != "." and piece.islower()

    def same_side(self, a, b):
        if a == "." or b == ".":
            return False

        return (
            (a.isupper() and b.isupper()) or
            (a.islower() and b.islower())
        )

    def enemy_side(self, a, b):
        return (
            a != "." and
            b != "." and
            not self.same_side(a, b)
        )

    def directions(self, piece):
        p = piece.lower()

        if p == "r":
            return [(1,0),(-1,0),(0,1),(0,-1)]

        if p == "b":
            return [(1,1),(1,-1),(-1,1),(-1,-1)]

        if p == "q":
            return [
                (1,0),(-1,0),(0,1),(0,-1),
                (1,1),(1,-1),(-1,1),(-1,-1)
            ]

        return []

    # --------------------------------------------------
    # ATTACK DETECTION
    # --------------------------------------------------

    def square_attacked(self, r, c, by_color):

        # Pawn attacks
        if by_color == "white":
            pawn_row = r + 1

            for dc in (-1, 1):
                pc = c + dc

                if self.inside(pawn_row, pc):
                    if self.board[pawn_row][pc] == "P":
                        return True

        else:
            pawn_row = r - 1

            for dc in (-1, 1):
                pc = c + dc

                if self.inside(pawn_row, pc):
                    if self.board[pawn_row][pc] == "p":
                        return True

        # Knight attacks
        knight = "N" if by_color == "white" else "n"

        jumps = [
            (2,1),(2,-1),
            (-2,1),(-2,-1),
            (1,2),(1,-2),
            (-1,2),(-1,-2)
        ]

        for dr, dc in jumps:
            nr = r + dr
            nc = c + dc

            if self.inside(nr, nc):
                if self.board[nr][nc] == knight:
                    return True

        # King attacks
        king = "K" if by_color == "white" else "k"

        for dr in (-1,0,1):
            for dc in (-1,0,1):

                if dr == 0 and dc == 0:
                    continue

                nr = r + dr
                nc = c + dc

                if self.inside(nr, nc):
                    if self.board[nr][nc] == king:
                        return True

        # Sliding pieces
        enemy_rook = "R" if by_color == "white" else "r"
        enemy_bishop = "B" if by_color == "white" else "b"
        enemy_queen = "Q" if by_color == "white" else "q"

        # Rook / Queen
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:

            nr = r + dr
            nc = c + dc

            while self.inside(nr, nc):

                piece = self.board[nr][nc]

                if piece != ".":

                    if piece in (enemy_rook, enemy_queen):
                        return True

                    break

                nr += dr
                nc += dc

        # Bishop / Queen
        for dr, dc in [(1,1),(1,-1),(-1,1),(-1,-1)]:

            nr = r + dr
            nc = c + dc

            while self.inside(nr, nc):

                piece = self.board[nr][nc]

                if piece != ".":

                    if piece in (enemy_bishop, enemy_queen):
                        return True

                    break

                nr += dr
                nc += dc

        return False

    def find_king(self, color):

        king = "K" if color == "white" else "k"

        for r in range(8):
            for c in range(8):

                if self.board[r][c] == king:
                    return r, c

        return None

    def in_check(self, color):

        king = self.find_king(color)

        if king is None:
            return True

        opponent = "black" if color == "white" else "white"

        return self.square_attacked(
            king[0],
            king[1],
            opponent
        )

    # --------------------------------------------------
    # PSEUDO-LEGAL MOVES
    # --------------------------------------------------

    def piece_moves(self, r, c):

        piece = self.board[r][c]

        if piece == ".":
            return []

        if self.turn == "white" and not self.is_white(piece):
            return []

        if self.turn == "black" and not self.is_black(piece):
            return []

        moves = []

        p = piece.lower()

        # Pawn
        if p == "p":

            direction = -1 if piece.isupper() else 1
            start_row = 6 if piece.isupper() else 1

            nr = r + direction

            if self.inside(nr, c) and self.board[nr][c] == ".":

                moves.append((r,c,nr,c))

                nr2 = r + 2 * direction

                if (
                    r == start_row and
                    self.board[nr2][c] == "."
                ):
                    moves.append((r,c,nr2,c))

            # Captures
            for dc in (-1,1):

                nr = r + direction
                nc = c + dc

                if not self.inside(nr,nc):
                    continue

                target = self.board[nr][nc]

                if target != "." and not self.same_side(piece,target):
                    moves.append((r,c,nr,nc))

                # En passant
                if self.en_passant == (nr,nc):
                    moves.append((r,c,nr,nc))

            return moves

        # Knight
        if p == "n":

            jumps = [
                (2,1),(2,-1),
                (-2,1),(-2,-1),
                (1,2),(1,-2),
                (-1,2),(-1,-2)
            ]

            for dr,dc in jumps:

                nr = r + dr
                nc = c + dc

                if self.inside(nr,nc):

                    target = self.board[nr][nc]

                    if target == "." or not self.same_side(piece,target):
                        moves.append((r,c,nr,nc))

            return moves

        # King
        if p == "k":

            for dr in (-1,0,1):
                for dc in (-1,0,1):

                    if dr == 0 and dc == 0:
                        continue

                    nr = r + dr
                    nc = c + dc

                    if self.inside(nr,nc):

                        target = self.board[nr][nc]

                        if target == "." or not self.same_side(piece,target):
                            moves.append((r,c,nr,nc))

            # Castling
            if piece == "K":

                if self.castling["K"]:
                    if (
                        self.board[7][7] == "R" and
                        self.board[7][5] == "." and
                        self.board[7][6] == "." and
                        not self.in_check("white") and
                        not self.square_attacked(7,5,"black") and
                        not self.square_attacked(7,6,"black")
                    ):
                        moves.append((7,4,7,6))

                if self.castling["Q"]:
                    if (
                        self.board[7][0] == "R" and
                        self.board[7][1] == "." and
                        self.board[7][2] == "." and
                        self.board[7][3] == "." and
                        not self.in_check("white") and
                        not self.square_attacked(7,3,"black") and
                        not self.square_attacked(7,2,"black")
                    ):
                        moves.append((7,4,7,2))

            elif piece == "k":

                if self.castling["k"]:
                    if (
                        self.board[0][7] == "r" and
                        self.board[0][5] == "." and
                        self.board[0][6] == "." and
                        not self.in_check("black") and
                        not self.square_attacked(0,5,"white") and
                        not self.square_attacked(0,6,"white")
                    ):
                        moves.append((0,4,0,6))

                if self.castling["q"]:
                    if (
                        self.board[0][0] == "r" and
                        self.board[0][1] == "." and
                        self.board[0][2] == "." and
                        self.board[0][3] == "." and
                        not self.in_check("black") and
                        not self.square_attacked(0,3,"white") and
                        not self.square_attacked(0,2,"white")
                    ):
                        moves.append((0,4,0,2))

            return moves

        # Sliding pieces
        for dr,dc in self.directions(piece):

            nr = r + dr
            nc = c + dc

            while self.inside(nr,nc):

                target = self.board[nr][nc]

                if target == ".":
                    moves.append((r,c,nr,nc))

                else:

                    if not self.same_side(piece,target):
                        moves.append((r,c,nr,nc))

                    break

                nr += dr
                nc += dc

        return moves

    # --------------------------------------------------
    # MAKE MOVE ON COPY
    # --------------------------------------------------

    def copy(self):

        new = ChessEnvironment.__new__(ChessEnvironment)

        new.board = [row[:] for row in self.board]
        new.turn = self.turn
        new.move_count = self.move_count
        new.castling = self.castling.copy()
        new.en_passant = self.en_passant
        new.halfmove_clock = self.halfmove_clock
        new.position_history = self.position_history.copy()

        return new

    def apply_move_unchecked(self, move):

        r1,c1,r2,c2 = move

        piece = self.board[r1][c1]

        # En passant capture
        if (
            piece.lower() == "p" and
            self.en_passant == (r2,c2) and
            c1 != c2 and
            self.board[r2][c2] == "."
        ):
            capture_row = r2 + 1 if piece.isupper() else r2 - 1
            self.board[capture_row][c2] = "."

        self.board[r2][c2] = piece
        self.board[r1][c1] = "."

        # Castling rook movement
        if piece == "K":

            self.castling["K"] = False
            self.castling["Q"] = False

            if (r1,c1,r2,c2) == (7,4,7,6):
                self.board[7][5] = "R"
                self.board[7][7] = "."

            elif (r1,c1,r2,c2) == (7,4,7,2):
                self.board[7][3] = "R"
                self.board[7][0] = "."

        elif piece == "k":

            self.castling["k"] = False
            self.castling["q"] = False

            if (r1,c1,r2,c2) == (0,4,0,6):
                self.board[0][5] = "r"
                self.board[0][7] = "."

            elif (r1,c1,r2,c2) == (0,4,0,2):
                self.board[0][3] = "r"
                self.board[0][0] = "."

        # Rook movement disables castling
        if (r1,c1) == (7,0):
            self.castling["Q"] = False

        if (r1,c1) == (7,7):
            self.castling["K"] = False

        if (r1,c1) == (0,0):
            self.castling["q"] = False

        if (r1,c1) == (0,7):
            self.castling["k"] = False

        # Capturing rook disables castling
        if (r2,c2) == (7,0):
            self.castling["Q"] = False

        if (r2,c2) == (7,7):
            self.castling["K"] = False

        if (r2,c2) == (0,0):
            self.castling["q"] = False

        if (r2,c2) == (0,7):
            self.castling["k"] = False

        # Promotion
        if piece == "P" and r2 == 0:
            self.board[r2][c2] = "Q"

        elif piece == "p" and r2 == 7:
            self.board[r2][c2] = "q"

        # En passant target
        self.en_passant = None

        if piece.lower() == "p" and abs(r2-r1) == 2:
            self.en_passant = ((r1+r2)//2,c1)

    # --------------------------------------------------
    # LEGAL MOVES
    # --------------------------------------------------

    def legal_moves(self):

        legal = []

        for r in range(8):
            for c in range(8):

                for move in self.piece_moves(r,c):

                    # A king is never captured in chess.
                    # Checkmate ends the game before any such move
                    # can occur, so moves landing on the opponent king
                    # must never enter the legal action space.
                    r2, c2 = move[2], move[3]

                    if self.board[r2][c2].lower() == "k":
                        continue

                    test = self.copy()

                    test.apply_move_unchecked(move)

                    if not test.in_check(self.turn):
                        legal.append(move)

        return legal

    # --------------------------------------------------
    # EXECUTE MOVE
    # --------------------------------------------------

    def make_move(self, move):

        if move not in self.legal_moves():
            return False

        r1, c1, r2, c2 = move

        piece = self.board[r1][c1]
        target = self.board[r2][c2]

        is_pawn_move = piece.lower() == "p"

        is_capture = (
            target != "."
            or (
                is_pawn_move
                and self.en_passant == (r2, c2)
                and c1 != c2
            )
        )

        self.apply_move_unchecked(move)

        # 50-move rule clock.
        if is_pawn_move or is_capture:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        self.turn = (
            "black"
            if self.turn == "white"
            else "white"
        )

        self.move_count += 1

        self._record_position()

        return True

    def insufficient_material(self):
        """
        Detect common automatically-drawn dead positions.

        Covered:
        - King vs King
        - King + Bishop vs King
        - King + Knight vs King
        - King + Bishop vs King + Bishop
          when both bishops are on the same color squares
        """

        pieces = []

        for r in range(8):
            for c in range(8):

                piece = self.board[r][c]

                if piece != ".":
                    pieces.append(
                        (piece, r, c)
                    )

        non_kings = [
            item
            for item in pieces
            if item[0].lower() != "k"
        ]

        # King vs King
        if not non_kings:
            return True

        # Any pawn, rook or queen means mating material exists.
        for piece, _, _ in non_kings:
            if piece.lower() in ("p", "r", "q"):
                return False

        # King + single bishop/knight vs King
        if len(non_kings) == 1:
            return (
                non_kings[0][0].lower()
                in ("b", "n")
            )

        # King + bishop vs King + bishop.
        if len(non_kings) == 2:

            if all(
                piece.lower() == "b"
                for piece, _, _ in non_kings
            ):

                colors = []

                for _, r, c in non_kings:
                    colors.append(
                        (r + c) % 2
                    )

                return colors[0] == colors[1]

        return False

    def game_status(self):

        moves = self.legal_moves()

        if not moves:

            if self.in_check(self.turn):

                winner = (
                    "black"
                    if self.turn == "white"
                    else "white"
                )

                return f"checkmate:{winner}"

            return "stalemate"

        # 50-move rule.
        if self.halfmove_clock >= 100:
            return "draw:50_move"

        # Threefold repetition.
        if self.repetition_count() >= 3:
            return "draw:repetition"

        # Dead position / insufficient material.
        if self.insufficient_material():
            return "draw:insufficient_material"

        return (
            "check"
            if self.in_check(self.turn)
            else "playing"
        )

    def move_to_text(self, move):

        r1,c1,r2,c2 = move

        files = "abcdefgh"

        return (
            files[c1] + str(8-r1) +
            files[c2] + str(8-r2)
        )

    def display(self):

        print()
        print("    a b c d e f g h")
        print("   -----------------")

        for r,row in enumerate(self.board):
            print(f"{8-r} | " + " ".join(row) + f" | {8-r}")

        print("   -----------------")
        print("    a b c d e f g h")
        print()
        print(f"Turn: {self.turn}")
        print(f"Move: {self.move_count}")
        print(f"Status: {self.game_status()}")
        print()
