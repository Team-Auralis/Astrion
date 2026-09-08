class ChessActionSpace:
    """
    Fixed chess action space.

    Each action is:

        source square -> destination square

    64 source squares × 64 destination squares
    = 4096 possible actions.
    """

    SIZE = 4096

    @staticmethod
    def encode(move):
        r1, c1, r2, c2 = move

        source = r1 * 8 + c1
        target = r2 * 8 + c2

        return source * 64 + target

    @staticmethod
    def decode(action):
        if not 0 <= action < ChessActionSpace.SIZE:
            raise ValueError("Action must be between 0 and 4095")

        source = action // 64
        target = action % 64

        r1 = source // 8
        c1 = source % 8

        r2 = target // 8
        c2 = target % 8

        return (r1, c1, r2, c2)

    @staticmethod
    def move_to_uci(move):
        r1, c1, r2, c2 = move

        files = "abcdefgh"

        return (
            files[c1] + str(8 - r1) +
            files[c2] + str(8 - r2)
        )

    @staticmethod
    def action_to_uci(action):
        return ChessActionSpace.move_to_uci(
            ChessActionSpace.decode(action)
        )

    @staticmethod
    def legal_action_mask(environment):
        """
        Returns a fixed 4096-element list.

        1 = legal action
        0 = illegal action
        """

        mask = [0] * ChessActionSpace.SIZE

        for move in environment.legal_moves():
            action = ChessActionSpace.encode(move)
            mask[action] = 1

        return mask
