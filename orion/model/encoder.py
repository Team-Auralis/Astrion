class ChessEncoder:
    """
    Converts a ChessEnvironment state into a numerical tensor.

    Shape:
        18 x 8 x 8

    Channels:
        0  = white pawn
        1  = white knight
        2  = white bishop
        3  = white rook
        4  = white queen
        5  = white king
        6  = black pawn
        7  = black knight
        8  = black bishop
        9  = black rook
        10 = black queen
        11 = black king
        12 = side to move
        13 = en-passant square
        14 = white kingside castling right
        15 = white queenside castling right
        16 = black kingside castling right
        17 = black queenside castling right
    """

    PIECE_CHANNELS = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11,
    }

    CHANNELS = 18

    def encode(self, state):
        tensor = [
            [
                [0 for _ in range(8)]
                for _ in range(8)
            ]
            for _ in range(self.CHANNELS)
        ]

        board = state["board"]

        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece in self.PIECE_CHANNELS:
                    channel = self.PIECE_CHANNELS[piece]
                    tensor[channel][row][col] = 1

        side_value = 1 if state["turn"] == "white" else 0
        for row in range(8):
            for col in range(8):
                tensor[12][row][col] = side_value

        en_passant = state.get("en_passant")
        if en_passant is not None:
            row, col = en_passant
            tensor[13][row][col] = 1

        castling = state.get("castling", {})
        if castling.get("K"):
            for row in range(8):
                for col in range(8):
                    tensor[14][row][col] = 1
        if castling.get("Q"):
            for row in range(8):
                for col in range(8):
                    tensor[15][row][col] = 1
        if castling.get("k"):
            for row in range(8):
                for col in range(8):
                    tensor[16][row][col] = 1
        if castling.get("q"):
            for row in range(8):
                for col in range(8):
                    tensor[17][row][col] = 1

        return tensor

    def flatten(self, tensor):
        return [
            value
            for channel in tensor
            for row in channel
            for value in row
        ]

    def encode_flat(self, state):
        return self.flatten(self.encode(state))

    def count_active(self, tensor):
        return sum(
            value
            for channel in tensor
            for row in channel
            for value in row
        )
