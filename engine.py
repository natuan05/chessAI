import torch
import chess as ch
from training import ChessCNN, model_eval


class Engine:
    def __init__(self, model, board, max_depth, color):
        self.board = board
        self.color = color
        self.max_depth = max_depth
        self.model = ChessCNN()
        self.model.load_state_dict(torch.load(model, weights_only=True))

    def get_best_move(self):
        return self._engine(None, 1)

    def _eval_function(self):
        evaluation = model_eval(self.model, self.board) + self._mate_opportunity()
        #for square in range(64):
        #    evaluation += self._square_res_points(ch.SQUARES[square])
        return evaluation

    def _mate_opportunity(self):
        if self.board.legal_moves.count() == 0:
            return -999 if self.board.turn == self.color else 999
        return 0

    def _square_res_points(self, square):
        piece_value = 0
        piece_type = self.board.piece_type_at(square)

        if piece_type == ch.PAWN:
            piece_value = 1
        elif piece_type == ch.ROOK:
            piece_value = 5.1
        elif piece_type == ch.BISHOP:
            piece_value = 3.33
        elif piece_type == ch.KNIGHT:
            piece_value = 3.2
        elif piece_type == ch.QUEEN:
            piece_value = 8.8

        return -piece_value if self.board.color_at(square) != self.color else piece_value

    def _engine(self, candidate, depth):
        if depth == self.max_depth or self.board.legal_moves.count() == 0:
            return self._eval_function()

        best_move_value = float("-inf") if depth % 2 != 0 else float("inf")
        move = None

        for move_candidate in self.board.legal_moves:
            self.board.push(move_candidate)
            value = self._engine(best_move_value, depth + 1)

            if (depth % 2 != 0 and value > best_move_value) or (depth % 2 == 0 and value < best_move_value):
                best_move_value = value
                if depth == 1:
                    move = move_candidate

            if candidate is not None and ((depth % 2 == 0 and value < candidate) or (depth % 2 != 0 and value > candidate)):
                self.board.pop()
                break

            self.board.pop()

        return move if depth == 1 else best_move_value
