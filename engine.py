import torch
import chess as ch
from training import ChessCNN, model_eval
from copy import deepcopy  # 🆕 Import để sao chép board

class Engine:
    def __init__(self, model, board, max_depth, color):
        self.original_board = board  # 🧠 giữ lại board thật
        self.color = color
        self.max_depth = max_depth
        self.model = ChessCNN()
        self.model.load_state_dict(torch.load(model, weights_only=True))

    def get_best_move(self):
        # 🆕 tạo bản copy của board cho quá trình duyệt
        copied_board = deepcopy(self.original_board)
        return self._engine(copied_board, None, 1)

    def _eval_function(self, board):
        evaluation = model_eval(self.model, board) + self._mate_opportunity(board)
        return evaluation

    def _mate_opportunity(self, board):
        if board.legal_moves.count() == 0:
            return -999 if board.turn == self.color else 999
        return 0

    def _engine(self, board, candidate, depth):
        if depth == self.max_depth or board.legal_moves.count() == 0:
            return self._eval_function(board)

        best_move_value = float("-inf") if depth % 2 != 0 else float("inf")
        move = None

        for move_candidate in board.legal_moves:
            board.push(move_candidate)
            value = self._engine(board, best_move_value, depth + 1)

            if (depth % 2 != 0 and value > best_move_value) or (depth % 2 == 0 and value < best_move_value):
                best_move_value = value
                if depth == 1:
                    move = move_candidate

            board.pop()

            if candidate is not None:
                if (depth % 2 == 0 and value < candidate) or (depth % 2 != 0 and value > candidate):
                    break  # cắt nhánh alpha-beta đơn giản

        return move if depth == 1 else best_move_value
